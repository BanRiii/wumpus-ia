"""
IA cazadora para Wumpus World (grafo no dirigido + Wumpus dinámico).

NO modifica wumpus_world.py: solo usa su API pública
(perceptions_at, move_player, shoot_arrow, grab_gold, climb_out,
maybe_move_wumpus) y lee la configuración (WorldConfig).

Qué sabe el agente (información legítima del jugador):
    - El mapa completo (el grafo se ve en pantalla).
    - Las reglas: nº de pozos, flechas, y cómo despierta el Wumpus
      (probabilidad y intervalo forzado).
    - Sus percepciones en la sala actual: hedor / brisa / brillo,
      y si su flecha mató o no al Wumpus (el "grito").
  NO sabe dónde está el Wumpus, ni los pozos, ni el oro, ni cuándo
  se mueve el Wumpus.

Cómo razona:
    1. Pozos (estáticos): enumera TODAS las configuraciones de pozos
       compatibles con las brisas observadas -> probabilidad exacta
       de pozo por sala.
    2. Wumpus (dinámico): filtro bayesiano discreto sobre las salas.
         - Predicción: tras cada turno el Wumpus se mueve con la
           probabilidad que fijan las reglas, a un vecino uniforme.
         - Corrección: el hedor en la sala actual implica que el
           Wumpus está en un vecino (y su ausencia, que NO lo está).
         - Una flecha fallida descarta esa sala.
    3. Decide: dispara cuando la creencia concentrada en una sala
       adyacente supera un umbral; si no, se mueve (solo por salas
       de riesgo ~0) hacia el punto que más información da, o pasa
       turno. Con el Wumpus muerto, busca el oro y sale.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from wumpus_world import CaveGraph, WorldConfig, WumpusWorld, build_classic_cave


# --------------------------------------------------------------------------
# Acciones y reportes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    kind: str                    # "move" | "shoot" | "grab" | "climb" | "wait"
    room: Optional[int] = None
    why: str = ""                # explicación legible (para logs / GUI)


@dataclass
class StepReport:
    action: Action
    message: str
    wumpus_move: Optional[tuple] = None   # (vieja, nueva, hedor_removido, hedor_agregado)


def _binary_entropy(p: float) -> float:
    if p <= 1e-12 or p >= 1 - 1e-12:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# --------------------------------------------------------------------------
# Agente
# --------------------------------------------------------------------------
class HunterAgent:
    def __init__(
        self,
        graph: CaveGraph,
        config: Optional[WorldConfig] = None,
        *,
        shoot_threshold: float = 0.80,   # confianza para disparar de inmediato
        min_threshold: float = 0.50,     # umbral al que decae la paciencia
        decay_per_idle_turn: float = 0.005,
        max_move_risk: float = 0.03,     # riesgo tolerado en movimientos "seguros"
        loot_risk_cap: float = 0.25,     # riesgo máximo que acepta al buscar el oro / romper bloqueos
        wait_limit: int = 150,           # turnos de espera segura antes de arriesgarse
        stuck_shot_min: float = 0.75,    # confianza para disparar si no hay salida segura...
        stuck_decay: float = 0.03,       # ...que baja con cada turno de espera (mín. 0.30)
        seed: Optional[int] = None,
    ) -> None:
        self.g = graph
        self.cfg = config or WorldConfig()
        self.rooms = graph.nodes()
        self.nb = {r: sorted(graph.neighbors(r)) for r in self.rooms}
        self.entrance = self.cfg.entrance_room
        self.rng = random.Random(seed)

        self.shoot_threshold = shoot_threshold
        self.min_threshold = min_threshold
        self.decay = decay_per_idle_turn
        self.max_move_risk = max_move_risk
        self.loot_risk_cap = loot_risk_cap
        self.wait_limit = wait_limit
        self.stuck_shot_min = stuck_shot_min
        self.stuck_decay = stuck_decay
        self.consec_waits = 0

        # Conocimiento / estado
        self.pos = self.entrance
        self.turn = 0
        self.arrows = self.cfg.num_arrows
        self.has_gold = False
        self.wumpus_alive = True
        self.visited: set[int] = set()
        self.breeze: dict[int, bool] = {}
        self.pit_prob: dict[int, float] = {r: 0.0 for r in self.rooms}
        self.idle_turns = 0              # turnos de caza sin disparar
        self.last_room: Optional[int] = None

        # Creencia inicial sobre el Wumpus: uniforme salvo la entrada
        self.belief: dict[int, float] = {
            r: (0.0 if r == self.entrance else 1.0) for r in self.rooms
        }
        self._normalize()
        self._recompute_pits()

    # ------------------------------------------------------------------
    # Inferencia de pozos (exacta, por enumeración)
    # ------------------------------------------------------------------
    def _recompute_pits(self) -> None:
        cands = [r for r in self.rooms if r != self.entrance and r not in self.visited]
        index = {r: i for i, r in enumerate(cands)}
        obs_masks: list[tuple[int, bool]] = []
        for room, breeze in self.breeze.items():
            mask = 0
            for n in self.nb[room]:
                if n in index:
                    mask |= 1 << index[n]
            obs_masks.append((mask, breeze))

        k = self.cfg.num_pits
        counts = [0] * len(cands)
        total = 0
        for combo in itertools.combinations(range(len(cands)), k):
            m = 0
            for i in combo:
                m |= 1 << i
            if all(bool(m & mask) == b for mask, b in obs_masks):
                total += 1
                for i in combo:
                    counts[i] += 1

        self.pit_prob = {r: 0.0 for r in self.rooms}
        if total == 0:               # modelo inconsistente: ser conservador
            for r in cands:
                self.pit_prob[r] = 0.5
            return
        for r, i in index.items():
            self.pit_prob[r] = counts[i] / total

    # ------------------------------------------------------------------
    # Filtro bayesiano del Wumpus
    # ------------------------------------------------------------------
    def _normalize(self) -> None:
        s = sum(self.belief.values())
        if s <= 0:
            n = len(self.rooms)
            self.belief = {r: 1.0 / n for r in self.rooms}
        else:
            self.belief = {r: v / s for r, v in self.belief.items()}

    def _p_move(self, turn: int) -> float:
        """Probabilidad de que el Wumpus se mueva al cerrar el turno 'turn'."""
        forced = turn > 0 and turn % self.cfg.wumpus_move_interval == 0
        return 1.0 if forced else self.cfg.wumpus_wake_prob

    def _predict(self, belief: dict[int, float], turn: int) -> dict[int, float]:
        p = self._p_move(turn)
        new = {r: (1 - p) * b for r, b in belief.items()}
        for w, b in belief.items():
            if b <= 0:
                continue
            share = p * b / len(self.nb[w])
            for y in self.nb[w]:
                new[y] += share
        return new

    def _observe_stench(self, room: int, stench: bool) -> None:
        near = set(self.nb[room])
        eps = 1e-6                    # tolerancia a un modelo imperfecto
        for w in self.rooms:
            ok = (w in near) == stench
            self.belief[w] *= 1.0 if ok else eps
        self._normalize()

    # ------------------------------------------------------------------
    # Ciclo percepción -> acción -> retroalimentación
    # ------------------------------------------------------------------
    def choose_action(self, room: int, percepts: dict[str, bool]) -> Action:
        self.pos = room
        first_time = room not in self.visited
        self.visited.add(room)
        self.breeze[room] = percepts["brisa"]
        if first_time:
            self._recompute_pits()
        if self.wumpus_alive:
            self._observe_stench(room, percepts["hedor"])

        if percepts["brillo"] and not self.has_gold:
            return Action("grab", why="Brillo: hay oro aquí")

        if self.wumpus_alive and self.arrows > 0:
            action = self._hunt_step()
            if action is not None:
                return action
        return self._loot_exit_step()

    def notify_outcome(self, action: Action, world: WumpusWorld) -> None:
        """Retroalimentación tras ejecutar la acción (antes de que el turno cierre)."""
        self.consec_waits = self.consec_waits + 1 if action.kind == "wait" else 0
        if action.kind == "shoot":
            self.arrows = world.arrows
            self.idle_turns = 0
            if not world.wumpus_alive:
                self.wumpus_alive = False          # el "grito"
            else:                                   # fallo: no estaba ahí
                self.belief[action.room] = 1e-9
                self._normalize()
        elif action.kind == "grab":
            self.has_gold = world.has_gold
        elif action.kind == "move":
            self.last_room = self.pos
            self.pos = action.room
            if self.wumpus_alive:                   # sobrevivimos: no estaba ahí
                self.belief[action.room] = 1e-9
                self._normalize()

    def end_turn(self, turn: int) -> None:
        """Se llama tras cerrar el turno (aquí el Wumpus pudo moverse)."""
        self.turn = turn
        if self.wumpus_alive:
            self.belief = self._predict(self.belief, turn)
            self._normalize()
            if self.arrows > 0:
                self.idle_turns += 1

    # ------------------------------------------------------------------
    # Riesgo
    # ------------------------------------------------------------------
    def room_risk(self, r: int) -> float:
        pw = self.belief[r] if self.wumpus_alive else 0.0
        return 1.0 - (1.0 - self.pit_prob[r]) * (1.0 - pw)

    # ------------------------------------------------------------------
    # Fase de caza
    # ------------------------------------------------------------------
    def _threshold(self) -> float:
        return max(self.min_threshold, self.shoot_threshold - self.decay * self.idle_turns)

    def _hunt_step(self) -> Optional[Action]:
        pos = self.pos
        nbrs = self.nb[pos]
        shot = max(nbrs, key=lambda y: self.belief[y])
        p_shot = self.belief[shot]

        if p_shot >= self._threshold():
            return Action("shoot", shot, f"P(Wumpus en {shot}) = {p_shot:.0%}")

        # Opciones de movimiento "seguras"
        options: list[tuple[float, Action]] = []
        for x in nbrs:
            if self.room_risk(x) > self.max_move_risk:
                continue
            b = dict(self.belief)
            b[x] = 1e-9
            s = sum(b.values())
            b = {r: v / s for r, v in b.items()}
            bp = self._predict(b, self.turn + 1)
            p_stench = sum(bp[w] for w in self.nb[x])
            score = _binary_entropy(p_stench) + 0.02 * self.rng.random()
            if x == self.last_room:
                score -= 0.01           # leve preferencia por no repetir
            options.append((score, Action("move", x, f"Patrulla a {x} (riesgo {self.room_risk(x):.1%})")))

        if options:
            options.sort(key=lambda t: t[0], reverse=True)
            return options[0][1]

        # Sin movimientos seguros: disparar a la mejor apuesta, esperar o (tras
        # agotar la paciencia) dejar que la fase de exploración decida si arriesgarse.
        stuck_thr = max(0.30, self.stuck_shot_min - self.stuck_decay * self.consec_waits)
        if p_shot >= stuck_thr:
            return Action("shoot", shot, f"Sin salida segura; P(Wumpus en {shot}) = {p_shot:.0%}")
        if self.consec_waits < self.wait_limit:
            return Action("wait", why="Sin movimientos seguros: espero")
        return None

    # ------------------------------------------------------------------
    # Fase de oro / salida
    # ------------------------------------------------------------------
    def _safe_bfs(self, start: int, goal_test: Callable[[int], bool], risk_cap: float) -> Optional[list[int]]:
        prev = {start: None}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if u != start and goal_test(u):
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if u != start and u not in self.visited:
                continue                 # no atravesar salas desconocidas
            for v in self.nb[u]:
                if v not in prev and self.room_risk(v) <= risk_cap:
                    prev[v] = u
                    dq.append(v)
        return None

    def _loot_exit_step(self) -> Action:
        pos = self.pos
        if self.has_gold:
            return self._go_exit("Con el oro: regreso a la salida")

        # 1) sala desconocida con riesgo ~0
        path = self._safe_bfs(pos, lambda r: r not in self.visited, self.max_move_risk)
        if path:
            return Action("move", path[1], f"Exploro {path[-1]} (riesgo {self.room_risk(path[-1]):.1%})")

        # 2) apostar a la sala desconocida de menor riesgo
        best = None
        for v in self.visited:
            for r in self.nb[v]:
                if r in self.visited:
                    continue
                risk = self.room_risk(r)
                p = self._safe_bfs(pos, lambda x, v=v: x == v, self.max_move_risk) if v != pos else [pos]
                if p is None:
                    continue
                if best is None or risk < best[0]:
                    best = (risk, r, p)
        if best and best[0] <= self.loot_risk_cap:
            risk, target, p = best
            nxt = p[1] if len(p) > 1 else target
            return Action("move", nxt, f"Apuesta por {target} (riesgo {risk:.0%})")

        return self._go_exit("Sin salas seguras que explorar: salgo")

    def _go_exit(self, why: str) -> Action:
        if self.pos == self.entrance:
            return Action("climb", why=why)
        path = self._safe_bfs(self.pos, lambda r: r == self.entrance, 1.0)
        if path:
            return Action("move", path[1], why)
        return Action("wait", why="Sin ruta de regreso")


# --------------------------------------------------------------------------
# Ejecución de acciones sobre el juego (usa solo la API pública del mundo)
# --------------------------------------------------------------------------
def apply_action(world: WumpusWorld, action: Action) -> str:
    if action.kind == "move":
        return world.move_player(action.room)
    if action.kind == "shoot":
        return world.shoot_arrow(action.room)
    if action.kind == "grab":
        return world.grab_gold()
    if action.kind == "climb":
        return world.climb_out()
    if action.kind == "wait":
        return "La IA pasa el turno."
    raise ValueError(f"Acción desconocida: {action}")


def play_step(world: WumpusWorld, agent: HunterAgent) -> StepReport:
    """Un turno completo: percibir -> decidir -> actuar -> cerrar turno."""
    percepts = world.perceptions_at(world.player_room)
    action = agent.choose_action(world.player_room, percepts)
    message = apply_action(world, action)
    agent.notify_outcome(action, world)

    report = StepReport(action, message)
    if not world.game_over:
        world.turn += 1
        report.wumpus_move = world.maybe_move_wumpus()
        agent.end_turn(world.turn)
    return report


@dataclass
class EpisodeResult:
    killed_wumpus: bool
    won: bool
    died: bool
    cause: str
    steps: int


def run_episode(world: WumpusWorld, agent: HunterAgent, max_steps: int = 600,
                on_step: Optional[Callable[[StepReport], None]] = None) -> EpisodeResult:
    steps = 0
    last_msg = ""
    while not world.game_over and steps < max_steps:
        rep = play_step(world, agent)
        last_msg = rep.message
        steps += 1
        if on_step:
            on_step(rep)
    died = world.game_over and not world.won and ("Fin del juego" in last_msg)
    cause = ""
    if died:
        cause = "pozo" if "pozo" in last_msg else "wumpus"
    elif not world.game_over:
        cause = "límite de pasos"
    return EpisodeResult(not world.wumpus_alive, world.won, died, cause, steps)


# --------------------------------------------------------------------------
# Generación de mundos
# --------------------------------------------------------------------------
def make_world(graph: CaveGraph, config: Optional[WorldConfig] = None,
               fair_start: bool = False) -> WumpusWorld:
    """
    fair_start=True: re-sortea el mundo hasta que ningún vecino de la
    entrada sea un pozo (así la IA no depende de la suerte pura en su
    primer movimiento). No toca la lógica de WumpusWorld.
    """
    cfg = config or WorldConfig()
    while True:
        world = WumpusWorld(graph, cfg)
        if not fair_start:
            return world
        if not (world.pit_rooms & graph.neighbors(cfg.entrance_room)):
            return world


# --------------------------------------------------------------------------
# Benchmark / modo texto
# --------------------------------------------------------------------------
def benchmark(n: int, fair_start: bool, seed: int = 0, **agent_kw) -> dict:
    random.seed(seed)
    cfg = WorldConfig()
    killed = won = died = timeouts = 0
    by_cause = {"pozo": 0, "wumpus": 0}
    total_steps = 0
    for i in range(n):
        graph = build_classic_cave()
        world = make_world(graph, cfg, fair_start)
        agent = HunterAgent(graph, cfg, seed=seed * 100003 + i, **agent_kw)
        res = run_episode(world, agent)
        killed += res.killed_wumpus
        won += res.won
        died += res.died
        timeouts += (res.cause == "límite de pasos")
        if res.died:
            by_cause[res.cause] += 1
        total_steps += res.steps
    return {
        "n": n, "kill": killed / n, "win": won / n, "death": died / n,
        "death_pit": by_cause["pozo"] / n, "death_wumpus": by_cause["wumpus"] / n,
        "timeout": timeouts / n, "avg_steps": total_steps / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="IA cazadora de Wumpus (modo texto / benchmark)")
    ap.add_argument("--bench", type=int, default=0, help="nº de partidas para estadística")
    ap.add_argument("--fair", action="store_true", help="mundos con entrada sin pozos adyacentes")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.bench:
        r = benchmark(args.bench, args.fair, seed=args.seed or 0)
        print(r)
        return

    if args.seed is not None:
        random.seed(args.seed)
    cfg = WorldConfig()
    graph = build_classic_cave()
    world = make_world(graph, cfg, args.fair)
    agent = HunterAgent(graph, cfg, seed=args.seed)

    def show(rep: StepReport) -> None:
        print(f"[t={world.turn:3d}] {rep.action.kind:5s} {rep.action.room or '':>3} | {rep.action.why}")
        print(f"        -> {rep.message}")
        if rep.wumpus_move:
            print(f"        (el Wumpus se movió {rep.wumpus_move[0]} -> {rep.wumpus_move[1]})")

    res = run_episode(world, agent, on_step=show)
    print("\nWumpus cazado:", res.killed_wumpus, "| Victoria:", res.won, "| Pasos:", res.steps)


if __name__ == "__main__":
    main()
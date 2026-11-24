"""
Modo "IA juega sola" para tu interfaz de Pygame.

No modifica wumpus_world.py, wumpus_gui.py ni wumpus_ai.py: define una
subclase de WumpusGUI que, en cada tick de un temporizador, pide un
turno a la IA (play_step) y deja que la interfaz lo dibuje.

Controles:
    ESPACIO  pausa / reanuda a la IA
    N        un solo turno (estando en pausa)
    + / -    más rápido / más lento
    B        mostrar / ocultar la creencia de la IA sobre el Wumpus (anillos rojos)
    X        mostrar / ocultar la posición REAL del Wumpus (modo rayos X)
    ESC      salir

Uso:
    python wumpus_gui_ai.py            # mundo aleatorio
    python wumpus_gui_ai.py --fair     # entrada sin pozos adyacentes
"""

from __future__ import annotations

import argparse

import pygame

from wumpus_ai import HunterAgent, make_world, play_step
from wumpus_gui import WumpusGUI
from wumpus_world import WorldConfig, WumpusWorld, build_classic_cave


class AutoWumpusGUI(WumpusGUI):
    """WumpusGUI + una IA que juega sola. Solo añade control de la IA y overlays."""

    def __init__(self, world: WumpusWorld, agent: HunterAgent, step_ms: int = 600, **kw) -> None:
        super().__init__(world, **kw)
        pygame.display.set_caption("Wumpus World - IA cazadora")
        self.agent = agent
        self.step_ms = step_ms
        self.paused = False
        self.show_belief = True
        self.show_wumpus = False
        self._last_step = pygame.time.get_ticks()
        self._why = "Esperando..."

    # ------------------------------------------------------------------
    # Un turno de la IA, volcado al log y al estado visual de la GUI
    # ------------------------------------------------------------------
    def _ai_step(self) -> None:
        world = self.world
        if world.game_over:
            return
        rep = play_step(world, self.agent)
        self._why = rep.action.why
        self.visited.add(world.player_room)

        label = {"move": f"mover a {rep.action.room}", "shoot": f"disparar a {rep.action.room}",
                 "grab": "agarrar oro", "climb": "salir", "wait": "esperar"}[rep.action.kind]
        self._log(f"IA: {label}")
        if rep.action.kind in ("shoot", "grab", "climb") or world.game_over:
            self._log(rep.message)
        if rep.wumpus_move is not None:
            old_room, new_room, removed, added = rep.wumpus_move
            self._log(f"El Wumpus se movió: {old_room} -> {new_room}")

    # ------------------------------------------------------------------
    # Bucle principal (reemplaza al de la clase base para añadir el temporizador)
    # ------------------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_n and self.paused:
                        self._ai_step()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.step_ms = max(40, int(self.step_ms * 0.7))
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.step_ms = min(3000, int(self.step_ms / 0.7))
                    elif event.key == pygame.K_b:
                        self.show_belief = not self.show_belief
                    elif event.key == pygame.K_x:
                        self.show_wumpus = not self.show_wumpus

            now = pygame.time.get_ticks()
            if not self.paused and not self.world.game_over and now - self._last_step >= self.step_ms:
                self._last_step = now
                self._ai_step()

            self._draw()
            self.clock.tick(30)

        pygame.quit()

    # ------------------------------------------------------------------
    # Overlays: se dibujan encima de los nodos que ya pinta la clase base
    # ------------------------------------------------------------------
    def _draw_nodes(self) -> None:
        super()._draw_nodes()
        world = self.world

        if self.show_belief and world.wumpus_alive:
            for node, p in self.agent.belief.items():
                if p < 0.02:
                    continue
                x, y = self.positions[node]
                radius = self.NODE_RADIUS + 3 + int(8 * min(p * 2, 1.0))
                width = 2 + int(4 * min(p * 2, 1.0))
                pygame.draw.circle(self.screen, (230, 70, 70), (int(x), int(y)), radius, width)

        if self.show_wumpus and world.wumpus_alive:
            x, y = self.positions[world.wumpus_room]
            pygame.draw.circle(self.screen, (255, 40, 40), (int(x), int(y)), self.NODE_RADIUS - 8)
            tag = self.small_font.render("W", True, (255, 255, 255))
            self.screen.blit(tag, tag.get_rect(center=(x, y)))

    def _draw_panel(self) -> None:
        super()._draw_panel()
        a = self.agent
        lines = [
            f"IA: {'PAUSA' if self.paused else 'jugando'}   velocidad: {self.step_ms} ms/turno",
            f"Razón: {self._why}",
            "ESPACIO pausa | N paso | +/- velocidad | B creencia | X rayos X",
        ]
        y = self.height - 24 * len(lines) - 10
        for line in lines:
            surf = self.small_font.render(line, True, (160, 220, 255))
            self.screen.blit(surf, (20, y))
            y += 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fair", action="store_true", help="entrada sin pozos adyacentes")
    ap.add_argument("--speed", type=int, default=600, help="milisegundos por turno")
    args = ap.parse_args()

    cfg = WorldConfig()
    graph = build_classic_cave()
    world = make_world(graph, cfg, fair_start=args.fair)
    agent = HunterAgent(graph, cfg)

    gui = AutoWumpusGUI(world, agent, step_ms=args.speed)   # <-- aquí se instancia
    gui.run()


if __name__ == "__main__":
    main()
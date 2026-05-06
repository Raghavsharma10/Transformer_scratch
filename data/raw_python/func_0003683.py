def timing_game(self):
        """Timing game."""
        self.ctrl_wg.game_timer.display(self.time)
        self.time += 1
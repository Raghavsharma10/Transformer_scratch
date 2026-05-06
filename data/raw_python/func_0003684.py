def reset_game(self):
        """Reset game board."""
        self.ms_game.reset_game()
        self.update_grid()
        self.time = 0
        self.timer.start(1000)
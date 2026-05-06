def reset(self):
        """
        Clears the `cells` and leaderboards, and sets all corners to `0,0`.
        """
        self.cells.clear()
        self.leaderboard_names.clear()
        self.leaderboard_groups.clear()
        self.top_left.set(0, 0)
        self.bottom_right.set(0, 0)
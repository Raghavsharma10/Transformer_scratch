def check_board(self):
        """Check the board status and give feedback."""
        num_mines = np.sum(self.info_map == 12)
        num_undiscovered = np.sum(self.info_map == 11)
        num_questioned = np.sum(self.info_map == 10)

        if num_mines > 0:
            return 0
        elif np.array_equal(self.info_map == 9, self.mine_map):
            return 1
        elif num_undiscovered > 0 or num_questioned > 0:
            return 2
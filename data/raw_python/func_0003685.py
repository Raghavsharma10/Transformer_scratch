def update_grid(self):
        """Update grid according to info map."""
        info_map = self.ms_game.get_info_map()
        for i in xrange(self.ms_game.board_height):
            for j in xrange(self.ms_game.board_width):
                self.grid_wgs[(i, j)].info_label(info_map[i, j])

        self.ctrl_wg.move_counter.display(self.ms_game.num_moves)
        if self.ms_game.game_status == 2:
            self.ctrl_wg.reset_button.setIcon(QtGui.QIcon(CONTINUE_PATH))
        elif self.ms_game.game_status == 1:
            self.ctrl_wg.reset_button.setIcon(QtGui.QIcon(WIN_PATH))
            self.timer.stop()
        elif self.ms_game.game_status == 0:
            self.ctrl_wg.reset_button.setIcon(QtGui.QIcon(LOSE_PATH))
            self.timer.stop()
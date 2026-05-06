def mousePressEvent(self, event):
        """Define mouse press event."""
        if event.button() == QtCore.Qt.LeftButton:
            # get label position
            p_wg = self.parent()
            p_layout = p_wg.layout()
            idx = p_layout.indexOf(self)
            loc = p_layout.getItemPosition(idx)[:2]
            if p_wg.ms_game.game_status == 2:
                p_wg.ms_game.play_move("click", loc[1], loc[0])
                p_wg.update_grid()
        elif event.button() == QtCore.Qt.RightButton:
            p_wg = self.parent()
            p_layout = p_wg.layout()
            idx = p_layout.indexOf(self)
            loc = p_layout.getItemPosition(idx)[:2]
            if p_wg.ms_game.game_status == 2:
                if self.id == 9:
                    self.info_label(10)
                    p_wg.ms_game.play_move("question", loc[1], loc[0])
                    p_wg.update_grid()
                elif self.id == 11:
                    self.info_label(9)
                    p_wg.ms_game.play_move("flag", loc[1], loc[0])
                    p_wg.update_grid()
                elif self.id == 10:
                    self.info_label(11)
                    p_wg.ms_game.play_move("unflag", loc[1], loc[0])
                    p_wg.update_grid()
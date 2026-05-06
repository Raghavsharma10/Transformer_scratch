def on_position_changed(self, position):
        """bind position changed signal with this"""
        if not self._lyric:
            return

        pos = find_previous(position*1000 + 300, self._pos_list)
        if pos is not None and pos != self._pos:
            self.current_sentence = self._pos_s_map[pos]
            self._pos = pos
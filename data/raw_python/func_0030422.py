def on_song_changed(self, song):
        """bind song changed signal with this"""
        if song is None or song.lyric is None:
            self._lyric = None
            self._pos_s_map = {}
        else:
            self._lyric = song.lyric.content
            self._pos_s_map = parse(self._lyric)
        self._pos_list = sorted(list(self._pos_s_map.keys()))
        self._pos = None
        self.current_sentence = ''
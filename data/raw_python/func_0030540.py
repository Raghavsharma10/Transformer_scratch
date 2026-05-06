def insert(self, song):
        """在当前歌曲后插入一首歌曲"""
        if song in self._songs:
            return
        if self._current_song is None:
            self._songs.append(song)
        else:
            index = self._songs.index(self._current_song)
            self._songs.insert(index + 1, song)
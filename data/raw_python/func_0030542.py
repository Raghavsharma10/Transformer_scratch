def current_song(self, song):
        """设置当前歌曲，将歌曲加入到播放列表，并发出 song_changed 信号

        .. note::

            该方法理论上只应该被 Player 对象调用。
        """
        self._last_song = self.current_song
        if song is None:
            self._current_song = None
        # add it to playlist if song not in playlist
        elif song in self._songs:
            self._current_song = song
        else:
            self.insert(song)
            self._current_song = song
        self.song_changed.emit(song)
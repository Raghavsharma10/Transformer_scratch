def play_song(self, song):
        """播放指定歌曲

        如果目标歌曲与当前歌曲不相同，则修改播放列表当前歌曲，
        播放列表会发出 song_changed 信号，player 监听到信号后调用 play 方法，
        到那时才会真正的播放新的歌曲。如果和当前播放歌曲相同，则忽略。

        .. note::

            调用方不应该直接调用 playlist.current_song = song 来切换歌曲
        """
        if song is not None and song == self.current_song:
            logger.warning('The song is already under playing.')
        else:
            self._playlist.current_song = song
def remove(self, song):
        """Remove song from playlist. O(n)

        If song is current song, remove the song and play next. Otherwise,
        just remove it.
        """
        if song in self._songs:
            if self._current_song is None:
                self._songs.remove(song)
            elif song == self._current_song:
                next_song = self.next_song
                # 随机模式下或者歌单只剩一首歌曲，下一首可能和当前歌曲相同
                if next_song == self.current_song:
                    self.current_song = None
                    self._songs.remove(song)
                    self.current_song = self.next_song
                else:
                    self.current_song = self.next_song
                    self._songs.remove(song)
            else:
                self._songs.remove(song)
            logger.debug('Remove {} from player playlist'.format(song))
        else:
            logger.debug('Remove failed: {} not in playlist'.format(song))

        if song in self._bad_songs:
            self._bad_songs.remove(song)
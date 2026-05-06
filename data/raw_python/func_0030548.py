def _on_song_changed(self, song):
        """播放列表 current_song 发生变化后的回调

        判断变化后的歌曲是否有效的，有效则播放，否则将它标记为无效歌曲。
        如果变化后的歌曲是 None，则停止播放。
        """
        logger.debug('Player received song changed signal')
        if song is not None:
            logger.info('Try to play song: %s' % song)
            if song.url:
                self.play(song.url)
            else:
                self._playlist.mark_as_bad(song)
                self.play_next()
        else:
            self.stop()
            logger.info('No good song in player playlist anymore.')
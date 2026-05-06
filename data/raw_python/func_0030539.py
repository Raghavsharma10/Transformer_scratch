def add(self, song):
        """往播放列表末尾添加一首歌曲"""
        if song in self._songs:
            return
        self._songs.append(song)
        logger.debug('Add %s to player playlist', song)
def _get_good_song(self, base=0, random_=False, direction=1):
        """从播放列表中获取一首可以播放的歌曲

        :param base: base index
        :param random: random strategy or not
        :param direction: forward if > 0 else backword

        >>> pl = Playlist([1, 2, 3])
        >>> pl._get_good_song()
        1
        >>> pl._get_good_song(base=1)
        2
        >>> pl._bad_songs = [2]
        >>> pl._get_good_song(base=1, direction=-1)
        1
        >>> pl._get_good_song(base=1)
        3
        >>> pl._bad_songs = [1, 2, 3]
        >>> pl._get_good_song()
        """
        if not self._songs or len(self._songs) <= len(self._bad_songs):
            logger.debug('No good song in playlist.')
            return None

        good_songs = []
        if direction > 0:
            song_list = self._songs[base:] + self._songs[0:base]
        else:
            song_list = self._songs[base::-1] + self._songs[:base:-1]
        for song in song_list:
            if song not in self._bad_songs:
                good_songs.append(song)
        if random_:
            return random.choice(good_songs)
        else:
            return good_songs[0]
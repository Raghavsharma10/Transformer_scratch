def fav_songs(self):
        """
        FIXME: 支持获取所有的收藏歌曲
        """
        if self._fav_songs is None:
            songs_data = self._api.user_favorite_songs(self.identifier)
            self._fav_songs = []
            if not songs_data:
                return
            for song_data in songs_data:
                song = _deserialize(song_data, NestedSongSchema)
                self._fav_songs.append(song)
        return self._fav_songs
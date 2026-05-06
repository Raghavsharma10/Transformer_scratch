def playlists(self):
        """获取用户创建的歌单

        如果不是用户本人，则不能获取用户默认精选集
        """
        if self._playlists is None:
            playlists_data = self._api.user_playlists(self.identifier)
            playlists = []
            for playlist_data in playlists_data:
                playlist = _deserialize(playlist_data, PlaylistSchema)
                playlists.append(playlist)
            self._playlists = playlists
        return self._playlists
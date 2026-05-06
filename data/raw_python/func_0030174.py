def _refresh_url(self):
        """刷新获取 url，失败的时候返回空而不是 None"""
        songs = self._api.weapi_songs_url([int(self.identifier)])
        if songs and songs[0]['url']:
            self.url = songs[0]['url']
        else:
            self.url = ''
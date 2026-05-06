def user_playlists(self, user_id, page=1, limit=30):
        """
        NOTE: 用户歌单有可能是仅自己可见
        """
        action = 'mtop.alimusic.music.list.collectservice.getcollectbyuser'
        payload = {
            'userId': user_id,
            'pagingVO': {
                'page': page,
                'pageSize': limit
            }
        }
        code, msg, rv = self.request(action, payload)
        # TODO: 支持获取更多
        return rv['data']['data']['collects']
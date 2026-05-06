def playlist_detail(self, playlist_id):
        """获取歌单详情

        如果歌单歌曲数超过 100 时，该接口的 songs 字段不会包含所有歌曲，
        但是它有个 allSongs 字段，会包含所有歌曲的 ID。
        """
        action = 'mtop.alimusic.music.list.collectservice.getcollectdetail'
        payload = {'listId': playlist_id}
        code, msg, rv = self.request(action, payload)
        return rv['data']['data']['collectDetail']
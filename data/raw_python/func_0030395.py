def update_playlist_song(self, playlist_id, song_id, op):
        """从播放列表删除或者增加一首歌曲

        如果歌曲不存在与歌单中，删除时返回 True；如果歌曲已经存在于
        歌单，添加时也返回 True。
        """
        action = 'mtop.alimusic.music.list.collectservice.{}songs'.format(
            'delete' if op == 'del' else 'add')
        payload = {
            'listId': playlist_id,
            'songIds': [song_id]
        }
        code, msg, rv = self.request(action, payload)
        return rv['data']['data']['success'] == 'true'
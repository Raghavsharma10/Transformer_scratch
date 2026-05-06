def op_music_to_playlist(self, mid, pid, op):
        """
        :param op: add or del
        """
        url_add = uri + '/playlist/manipulate/tracks'
        trackIds = '["' + str(mid) + '"]'
        data_add = {
            'tracks': str(mid),  # music id
            'pid': str(pid),    # playlist id
            'trackIds': trackIds,  # music id str
            'op': op   # opation
        }
        data = self.request('POST', url_add, data_add)
        code = data.get('code')

        # 从歌单中成功的移除歌曲时，code 是 200
        # 当从歌单中移除一首不存在的歌曲时，code 也是 200
        # 当向歌单添加歌曲时，如果歌曲已经在列表当中，
        # 返回 code 为 502
        if code == 200:
            return 1
        elif code == 502:
            return -1
        else:
            return 0
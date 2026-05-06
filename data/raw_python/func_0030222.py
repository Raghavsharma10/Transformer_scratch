def artist_detail(self, artist_id):
        """获取歌手详情"""
        path = '/v8/fcg-bin/fcg_v8_singer_track_cp.fcg'
        url = api_base_url + path
        params = {
            'singerid': artist_id,
            'songstatus': 1,
            'order': 'listen',
            'begin': 0,
            'num': 50,
            'from': 'h5',
            'platform': 'h5page',
        }
        resp = requests.get(url, params=params, timeout=self._timeout)
        rv = resp.json()
        return rv['data']
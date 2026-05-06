def search(self, s, stype=1, offset=0, total='true', limit=60):
        """get songs list from search keywords"""
        action = uri + '/search/get'
        data = {
            's': s,
            'type': stype,
            'offset': offset,
            'total': total,
            'limit': 60
        }
        resp = self.request('POST', action, data)
        if resp['code'] == 200:
            return resp['result']['songs']
        return []
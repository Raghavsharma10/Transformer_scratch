def album_infos(self, album_id):
        """
        :param album_id:
        :return: {
            code: int,
            album: { album }
        }
        """
        action = uri + '/album/' + str(album_id)
        data = self.request('GET', action)
        if data['code'] == 200:
            return data['album']
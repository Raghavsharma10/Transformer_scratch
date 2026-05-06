def is_connected(self, chassis):
        """Get Boolean connected status of the specified chassis."""
        self._check_session()
        try:
            status, data = self._rest.get_request('connections', chassis)
        except resthttp.RestHttpError as e:
            if int(e) == 404:
                # 404 NOT FOUND means the chassis in unknown, so return false.
                return False
        return bool(data and data.get('IsConnected'))
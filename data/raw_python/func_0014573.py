def join_session(self, sid):
        """Attach to an existing session."""
        self._rest.add_header('X-STC-API-Session', sid)
        self._sid = sid
        try:
            status, data = self._rest.get_request('objects', 'system1',
                                                  ['version', 'name'])
        except resthttp.RestHttpError as e:
            self._rest.del_header('X-STC-API-Session')
            self._sid = None
            raise RuntimeError('failed to join session "%s": %s' % (sid, e))

        return data['version']
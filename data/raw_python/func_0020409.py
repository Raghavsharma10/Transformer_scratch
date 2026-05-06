def homeautoswitch(self, cmd, ain=None, param=None):
        """
        Call a switch method.
        Should only be used by internal library functions.
        """
        assert self.sid, "Not logged in"
        params = {
            'switchcmd': cmd,
            'sid': self.sid,
        }
        if param is not None:
            params['param'] = param
        if ain:
            params['ain'] = ain

        url = self.base_url + '/webservices/homeautoswitch.lua'
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.text.strip().encode('utf-8')
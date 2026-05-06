def _get_attrib(cls):
        """Get matches element attributes."""
        if not cls._server_is_alive():
            cls._start_server_on_free_port()
        params = {'language': FAILSAFE_LANGUAGE, 'text': ''}
        data = urllib.parse.urlencode(params).encode()
        root = cls._get_root(cls._url, data, num_tries=1)
        return root.attrib
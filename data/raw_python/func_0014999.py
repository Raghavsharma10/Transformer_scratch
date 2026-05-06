def _get_languages(cls) -> set:
        """Get supported languages (by querying the server)."""
        if not cls._server_is_alive():
            cls._start_server_on_free_port()
        url = urllib.parse.urljoin(cls._url, 'Languages')
        languages = set()
        for e in cls._get_root(url, num_tries=1):
            languages.add(e.get('abbr'))
            languages.add(e.get('abbrWithVariant'))
        return languages
def _items(self):
        """
        Parse url and yield namedtuple Torrent for every torrent on page
        """
        torrents = map(self._get_torrent, self._get_rows())

        for t in torrents:
            yield t
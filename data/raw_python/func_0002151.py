def _get_data_raw(self):
        """Download observations matching the time range.

        Returns a tuple with a string for the body, string for the headers,
        and a list of dates.
        """
        # Import need to be here so we can monkeypatch urlopen for testing and avoid
        # downloading live data for testing
        try:
            from urllib.request import urlopen
        except ImportError:
            from urllib2 import urlopen

        with closing(urlopen(self.ftpsite + self.site_id + self.suffix + '.zip')) as url:
            f = ZipFile(BytesIO(url.read()), 'r').open(self.site_id + self.suffix)

        lines = [line.decode('utf-8') for line in f.readlines()]

        body, header, dates_long, dates = self._select_date_range(lines)

        return body, header, dates_long, dates
def set_error_pages(self, codes_map=None, common_prefix=None):
        """Add an error pages for managed 403, 404, 500 responses.

        Shortcut for ``.set_error_page()``.

        :param dict codes_map: Status code mapped into an html filepath or
            just a filename if common_prefix is used.

            If not set, filename containing status code is presumed: 400.html, 500.html, etc.

        :param str|unicode common_prefix: Common path (prefix) for all files.

        """
        statuses = [403, 404, 500]

        if common_prefix:
            if not codes_map:
                codes_map = {code: '%s.html' % code for code in statuses}

            for code, filename in codes_map.items():
                codes_map[code] = os.path.join(common_prefix, filename)

        for code, filepath in codes_map.items():
            self.set_error_page(code, filepath)

        return self._section
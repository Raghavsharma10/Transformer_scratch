def set_error_page(self, status, html_fpath):
        """Add an error page (html) for managed 403, 404, 500 response.

        :param int status: HTTP status code.

        :param str|unicode html_fpath: HTML page file path.

        """
        statuses = [403, 404, 500]

        status = int(status)

        if status not in statuses:
            raise ConfigurationError(
                'Code `%s` for `routing.set_error_page()` is unsupported. Supported: %s' %
                (status, ', '.join(map(str, statuses))))

        self._set('error-page-%s' % status, html_fpath, multi=True)

        return self._section
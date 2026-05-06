def iter_emails(self, number=-1, etag=None):
        """Iterate over email addresses for the authenticated user.

        :param int number: (optional), number of email addresses to return.
            Default: -1 returns all available email addresses
        :param str etag: (optional), ETag from a previous request to the same
            endpoint
        :returns: generator of dicts
        """
        url = self._build_url('user', 'emails')
        return self._iter(int(number), url, dict, etag=etag)
def cmd_connection_type(self):
        """Generates statistics on how many requests are made via HTTP and how
        many are made via SSL.

        .. note::
          This only works if the request path contains the default port for
          SSL (443).

        .. warning::
          The ports are hardcoded, they should be configurable.
        """
        https = 0
        non_https = 0
        for line in self._valid_lines:
            if line.is_https():
                https += 1
            else:
                non_https += 1
        return https, non_https
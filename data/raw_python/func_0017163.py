def _iter(self, count, url, cls, params=None, etag=None):
        """Generic iterator for this project.

        :param int count: How many items to return.
        :param int url: First URL to start with
        :param class cls: cls to return an object of
        :param params dict: (optional) Parameters for the request
        :param str etag: (optional), ETag from the last call
        """
        from .structs import GitHubIterator
        return GitHubIterator(count, url, cls, self, params, etag)
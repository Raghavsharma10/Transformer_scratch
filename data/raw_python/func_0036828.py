def construct_url(self):
        """Construct a full plex request URI, with `params`."""
        path = [self.path]
        path.extend([str(x) for x in self.params])

        url = self.client.base_url + '/'.join(x for x in path if x)
        query = self.kwargs.get('query')

        if query:
            # Dict -> List
            if type(query) is dict:
                query = query.items()

            # Remove items with `None` value
            query = [
                (k, v) for (k, v) in query
                if v is not None
            ]

            # Encode query, append to URL
            url += '?' + urlencode(query)

        return url
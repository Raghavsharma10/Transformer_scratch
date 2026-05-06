def get_many(self, uris):
        """
        Simple implementation,
        could be better implemented by backend not hitting db for every uri.
        """
        nodes = {}

        for uri in uris:
            try:
                node = self.get(uri)
            except NodeDoesNotExist:
                continue
            else:
                nodes[uri] = node

        return nodes
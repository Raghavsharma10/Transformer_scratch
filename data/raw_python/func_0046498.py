def delete_many(self, uris):
        """
        Simple implementation,
        could be better implemented by backend not hitting db for every uri.
        """
        deleted_nodes = {}

        for uri in uris:
            node = self.delete(uri)
            if node:
                deleted_nodes[uri] = node

        return deleted_nodes
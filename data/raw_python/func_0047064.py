def namespace_uri(self):
        """
        Finds and returns first applied URI of this node that has a namespace.

        :return str: uri
        """
        try:
            return next(
                iter(filter(lambda uri: URI(uri).namespace, self._uri))
            )
        except StopIteration:
            return None
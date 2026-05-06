def get_items(self):
        """
        Return the item models associated with this Publish group.
        """
        from .layers import Layer

        # no expansion support, just URLs
        results = []
        for url in self.items:
            if '/layers/' in url:
                r = self._client.request('GET', url)
                results.append(self._client.get_manager(Layer).create_from_result(r.json()))
            else:
                raise NotImplementedError("No support for %s" % url)
        return results
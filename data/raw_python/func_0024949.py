def get_collections(self):
        """
        Returns a flat list of the names of collections in the asset
        service.

        ..

            ['wind-turbines', 'jet-engines']

        """
        collections = []
        for result in self._get_collections():
            collections.append(result['collection'])

        return collections
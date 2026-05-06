def save(self, collection):
        """
        Save an asset collection to the service.
        """
        assert isinstance(collection, predix.data.asset.AssetCollection), "Expected AssetCollection"
        collection.validate()
        self.put_collection(collection.uri, collection.__dict__)
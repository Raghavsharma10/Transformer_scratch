def create_guid(self, collection=None):
        """
        Returns a new guid for use in posting a new asset to a collection.
        """
        guid = str(uuid.uuid4())
        if collection:
            return str.join('/', [collection, guid])
        else:
            return guid
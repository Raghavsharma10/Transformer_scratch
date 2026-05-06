def resource_collection_response(cls, offset=0, limit=20):
        """
        This method is deprecated for version 1.1.0.  Please use get_collection
        """
        request_args = {'page[offset]': offset, 'page[limit]': limit}
        return cls.get_collection(request_args)
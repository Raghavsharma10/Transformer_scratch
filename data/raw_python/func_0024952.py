def post_collection(self, collection, body):
        """
        Creates a new collection.  This is mostly just transport layer
        and passes collection and body along.  It presumes the body
        already has generated.

        The collection is *not* expected to have the id.
        """
        assert isinstance(body, (list)), "POST requires body to be a list"
        assert collection.startswith('/'), "Collections must start with /"
        uri = self.uri + '/v1' + collection
        return self.service._post(uri, body)
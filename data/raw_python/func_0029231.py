def get_collections(self):
        """ Get names of collections from request matchdict.

        :return: Names of collections
        :rtype: list of str
        """
        collections = self.request.matchdict['collections'].split('/')[0]
        collections = [coll.strip() for coll in collections.split(',')]
        return set(collections)
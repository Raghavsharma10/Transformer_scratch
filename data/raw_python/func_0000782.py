def update_many(self, **kwargs):
        """ Update multiple objects from collection.

        First ES is queried, then the results are used to query DB.
        This is done to make sure updated objects are those filtered
        by ES in the 'index' method (so user updates what he saw).
        """
        db_objects = self.get_dbcollection_with_es(**kwargs)
        return self.Model._update_many(
            db_objects, self._json_params, self.request)
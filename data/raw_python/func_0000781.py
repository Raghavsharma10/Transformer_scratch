def delete_many(self, **kwargs):
        """ Delete multiple objects from collection.

        First ES is queried, then the results are used to query the DB.
        This is done to make sure deleted objects are those filtered
        by ES in the 'index' method (so user deletes what he saw).
        """
        db_objects = self.get_dbcollection_with_es(**kwargs)
        return self.Model._delete_many(db_objects, self.request)
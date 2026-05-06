def get_dbcollection_with_es(self, **kwargs):
        """ Get DB objects collection by first querying ES. """
        es_objects = self.get_collection_es()
        db_objects = self.Model.filter_objects(es_objects)
        return db_objects
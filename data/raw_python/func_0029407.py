def get_collection_es(self):
        """ Query ES collection and return results.

        This is default implementation of querying ES collection with
        `self._query_params`. It must return found ES collection
        results for default response renderers to work properly.
        """
        from nefertari.elasticsearch import ES
        return ES(self.Model.__name__).get_collection(**self._query_params)
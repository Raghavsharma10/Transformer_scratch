def set_params(self, **params):
        """
        add/update solr query parameters
        """
        clone = copy.deepcopy(self)
        clone.adapter.set_params(**params)
        return clone
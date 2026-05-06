def base_object(self, data):
        """ Make sure to return all the existing filter fields
        for query results. """
        obj = {'id': data.get(self.id)}
        if self.parent is not None:
            obj['$parent'] = data.get(self.parent.id)
        return obj
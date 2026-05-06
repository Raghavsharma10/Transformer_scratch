def count(self, request, filter=None):
        """ Return the number of records matching the given filter. """
        if filter is None:
            filter = create_filter(request, self.mapped_class, self.geom_attr)
        query = self.Session().query(self.mapped_class)
        if filter is not None:
            query = query.filter(filter)
        return query.count()
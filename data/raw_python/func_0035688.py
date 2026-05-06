def _query(self, request, filter=None):
        """ Build a query based on the filter and the request params,
            and send the query to the database. """
        limit = None
        offset = None
        if 'maxfeatures' in request.params:
            limit = int(request.params['maxfeatures'])
        if 'limit' in request.params:
            limit = int(request.params['limit'])
        if 'offset' in request.params:
            offset = int(request.params['offset'])
        if filter is None:
            filter = create_filter(request, self.mapped_class, self.geom_attr)
        query = self.Session().query(self.mapped_class)
        if filter is not None:
            query = query.filter(filter)
        order_by = self._get_order_by(request)
        if order_by is not None:
            query = query.order_by(order_by)
        query = query.limit(limit).offset(offset)
        return query.all()
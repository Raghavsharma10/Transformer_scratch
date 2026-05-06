def read(self, request, filter=None, id=None):
        """ Build a query based on the filter or the idenfier, send the query
        to the database, and return a Feature or a FeatureCollection. """
        ret = None
        if id is not None:
            o = self.Session().query(self.mapped_class).get(id)
            if o is None:
                return HTTPNotFound()
            # FIXME: we return a Feature here, not a mapped object, do
            # we really want that?
            ret = self._filter_attrs(o.__geo_interface__, request)
        else:
            objs = self._query(request, filter)
            ret = FeatureCollection(
                [self._filter_attrs(o.__geo_interface__, request)
                 for o in objs if o is not None])
        return ret
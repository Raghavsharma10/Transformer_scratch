def update(self, request, id):
        """ Read the GeoJSON feature from the request body and update the
        corresponding object in the database. """
        if self.readonly:
            return HTTPMethodNotAllowed(headers={'Allow': 'GET, HEAD'})
        session = self.Session()
        obj = session.query(self.mapped_class).get(id)
        if obj is None:
            return HTTPNotFound()
        feature = loads(request.body, object_hook=GeoJSON.to_instance)
        if not isinstance(feature, Feature):
            return HTTPBadRequest()
        if self.before_update is not None:
            self.before_update(request, feature, obj)
        obj.__update__(feature)
        session.flush()
        request.response.status_int = 200
        return obj
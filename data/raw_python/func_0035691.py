def create(self, request):
        """ Read the GeoJSON feature collection from the request body and
            create new objects in the database. """
        if self.readonly:
            return HTTPMethodNotAllowed(headers={'Allow': 'GET, HEAD'})
        collection = loads(request.body, object_hook=GeoJSON.to_instance)
        if not isinstance(collection, FeatureCollection):
            return HTTPBadRequest()
        session = self.Session()
        objects = []
        for feature in collection.features:
            create = False
            obj = None
            if hasattr(feature, 'id') and feature.id is not None:
                obj = session.query(self.mapped_class).get(feature.id)
            if self.before_create is not None:
                self.before_create(request, feature, obj)
            if obj is None:
                obj = self.mapped_class(feature)
                create = True
            else:
                obj.__update__(feature)
            if create:
                session.add(obj)
            objects.append(obj)
        session.flush()
        collection = FeatureCollection(objects) if len(objects) > 0 else None
        request.response.status_int = 201
        return collection
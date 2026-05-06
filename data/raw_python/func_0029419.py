def _set_object_self(self, obj):
        """ Add '_self' key value to :obj: dict. """
        from nefertari.elasticsearch import ES
        location = self.request.path_url
        route_kwargs = {}

        """ Check for parents """
        if self.request.matchdict:
            route_kwargs.update(self.request.matchdict)
        try:
            type_, obj_pk = obj['_type'], obj['_pk']
        except KeyError:
            return
        resource = (self.model_collections.get(type_) or
                    self.model_collections.get(ES.src2type(type_)))
        if resource is not None:
            route_kwargs.update({resource.id_name: obj_pk})
            location = self.request.route_url(
                resource.uid, **route_kwargs)
        obj.setdefault('_self', location)
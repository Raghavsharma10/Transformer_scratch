def add_permission(self, resource, operation):
        '''Add a new :class:`Permission` for ``resource`` to perform an
``operation``. The resource can be either an object or a model.'''
        if isclass(resource):
            model_type = resource
            pk = ''
        else:
            model_type = resource.__class__
            pk = resource.pkvalue()
        p = Permission(model_type=model_type, object_pk=pk,
                       operation=operation)
        session = self.session
        if session.transaction:
            session.add(p)
            self.permissions.add(p)
            return p
        else:
            with session.begin() as t:
                t.add(p)
                self.permissions.add(p)
            return t.add_callback(lambda r: p)
def manager(self, model):
        '''Retrieve the :class:`Manager` for ``model`` which can be any of the
values valid for the :meth:`model` method.'''
        try:
            return self.router[model]
        except KeyError:
            meta = getattr(model, '_meta', model)
            if meta.type == 'structure':
                # this is a structure
                if hasattr(model, 'model'):
                    structure_model = model.model
                    if structure_model:
                        return self.manager(structure_model)
                    else:
                        manager = self.router.structure(model)
                        if manager:
                            return manager
            raise InvalidTransaction('"%s" not valid in this session' % meta)
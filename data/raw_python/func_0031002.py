def add_entity(self, domain, **kwargs):
        ''' Add a new Entity to tracking. '''

        # Set the entity's mapping func if one was specified
        map_func = kwargs.get('map_func', None)
        if map_func is not None and not callable(kwargs['map_func']):
            if self.entity_mapper is None:
                raise ValueError("Mapping function '%s' specified for Entity "
                                 "'%s', but no entity mapper was passed when "
                                 "initializing the current Layout. Please make"
                                 " sure the 'entity_mapper' argument is set." %
                                 (map_func, kwargs['name']))
            map_func = getattr(self.entity_mapper, kwargs['map_func'])
            kwargs['map_func'] = map_func

        ent = Entity(domain=domain, **kwargs)
        domain.add_entity(ent)

        if ent.mandatory:
            self.mandatory.add(ent.id)

        if ent.directory is not None:
            ent.directory = ent.directory.replace('{{root}}', self.root)

        self.entities[ent.id] = ent
        for alias in ent.aliases:
            self.entities[alias] = ent
        if self.dynamic_getters:
            func = partial(getattr(self, 'get'), target=ent.name,
                           return_type='id')
            func_name = inflect.engine().plural(ent.name)
            setattr(self, 'get_%s' % func_name, func)
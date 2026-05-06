def get_global_register_objects(self, do_sort=None, reverse=False, **kwargs):
        """Generate register objects (list) from register name list

        Usage: get_global_register_objects(name = ["Amp2Vbn", "GateHitOr", "DisableColumnCnfg"], address = [2, 3])
        Receives: keyword lists of register names, addresses,... for making cuts
        Returns: list of register objects

        """
        # speed up of the most often used keyword name
        try:
            names = iterable(kwargs.pop('name'))
        except KeyError:
            register_objects = []
        else:
            register_objects = [self.global_registers[reg] for reg in names]
        for keyword in kwargs.iterkeys():
            allowed_values = iterable(kwargs[keyword])
            register_objects.extend(filter(lambda global_register: set(iterable(global_register[keyword])).intersection(allowed_values), self.global_registers.itervalues()))
        if not register_objects and filter(None, kwargs.itervalues()):
            raise ValueError('Global register objects empty')
        if do_sort:
            return sorted(register_objects, key=itemgetter(*do_sort), reverse=reverse)
        else:
            return register_objects
def get_pixel_register_objects(self, do_sort=None, reverse=False, **kwargs):
        """Generate register objects (list) from register name list

        Usage: get_pixel_register_objects(name = ["TDAC", "FDAC"])
        Receives: keyword lists of register names, addresses,...
        Returns: list of register objects

        """
        # speed up of the most often used keyword name
        try:
            names = iterable(kwargs.pop('name'))
        except KeyError:
            register_objects = []
        else:
            register_objects = [self.pixel_registers[reg] for reg in names]
        for keyword in kwargs.iterkeys():
            allowed_values = iterable(kwargs[keyword])
            register_objects.extend(filter(lambda pixel_register: pixel_register[keyword] in allowed_values, self.pixel_registers.itervalues()))
        if not register_objects and filter(None, kwargs.itervalues()):
            raise ValueError('Pixel register objects empty')
        if do_sort:
            return sorted(register_objects, key=itemgetter(*do_sort), reverse=reverse)
        else:
            return register_objects
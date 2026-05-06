def get_global_register_attributes(self, register_attribute, do_sort=True, **kwargs):
        """Calculating register numbers from register names.

        Usage: get_global_register_attributes("attribute_name", name = [regname_1, regname_2, ...], addresses = 2)
        Receives: attribute name to be returned, dictionaries (kwargs) of register attributes and values for making cuts
        Returns: list of attribute values that matches dictionaries of attributes

        """
        # speed up of the most often used keyword name
        try:
            names = iterable(kwargs.pop('name'))
        except KeyError:
            register_attribute_list = []
        else:
            register_attribute_list = [self.global_registers[reg][register_attribute] for reg in names]
        for keyword in kwargs.keys():
            allowed_values = iterable(kwargs[keyword])
            try:
                register_attribute_list.extend(map(itemgetter(register_attribute), filter(lambda global_register: set(iterable(global_register[keyword])).intersection(allowed_values), self.global_registers.itervalues())))
            except AttributeError:
                pass
        if not register_attribute_list and filter(None, kwargs.itervalues()):
            raise ValueError('Global register attribute %s empty' % register_attribute)
        if do_sort:
            return sorted(set(flatten_iterable(register_attribute_list)))
        else:
            return flatten_iterable(register_attribute_list)
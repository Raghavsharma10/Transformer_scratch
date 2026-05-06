def get_attr_value(self, name):
        '''Retrieve the ``value`` for the attribute ``name``. The ``name``
can be nested following the :ref:`double underscore <tutorial-underscore>`
notation, for example ``group__name``. If the attribute is not available it
raises :class:`AttributeError`.'''
        if name in self._meta.dfields:
            return self._meta.dfields[name].get_value(self)
        elif not name.startswith('__') and JSPLITTER in name:
            bits = name.split(JSPLITTER)
            fname = bits[0]
            if fname in self._meta.dfields:
                return self._meta.dfields[fname].get_value(self, *bits[1:])
            else:
                return getattr(self, name)
        else:
            return getattr(self, name)
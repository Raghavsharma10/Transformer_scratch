def set_value(self, instance, value):
        '''Set the ``value`` for this :class:`Field` in a ``instance``
of a :class:`StdModel`.'''
        setattr(instance, self.attname, self.to_python(value))
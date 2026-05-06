def from_base64_data(cls, **kwargs):
        '''Load a :class:`StdModel` from possibly base64encoded data.

This method is used to load models from data obtained from the :meth:`tojson`
method.'''
        o = cls()
        meta = cls._meta
        pkname = meta.pkname()
        for name, value in iteritems(kwargs):
            if name == pkname:
                field = meta.pk
            elif name in meta.dfields:
                field = meta.dfields[name]
            else:
                continue
            value = field.to_python(value)
            setattr(o, field.attname, value)
        return o
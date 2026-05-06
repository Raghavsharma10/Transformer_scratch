def load_related_model(self, name, load_only=None, dont_load=None):
        '''Load a the :class:`ForeignKey` field ``name`` if this is part of the
fields of this model and if the related object is not already loaded.
It is used by the lazy loading mechanism of :ref:`one-to-many <one-to-many>`
relationships.

:parameter name: the :attr:`Field.name` of the :class:`ForeignKey` to load.
:parameter load_only: Optional parameters which specify the fields to load.
:parameter dont_load: Optional parameters which specify the fields not to load.
:return: the related :class:`StdModel` instance.
'''
        field = self._meta.dfields.get(name)
        if not field:
            raise ValueError('Field "%s" not available' % name)
        elif not field.type == 'related object':
            raise ValueError('Field "%s" not a foreign key' % name)
        return self._load_related_model(field, load_only, dont_load)
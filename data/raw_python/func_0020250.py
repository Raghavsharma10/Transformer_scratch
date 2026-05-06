def get_field(self, field):
        '''A :class:`Q` performs a series of operations and ultimately
generate of set of matched elements ``ids``. If on the other hand, a
different field is required, it can be specified with the :meth:`get_field`
method. For example, lets say a model has a field called ``object_id``
which contains ids of another model, we could use::

    qs = session.query(MyModel).get_field('object_id')

to obtain a set containing the values of matched elements ``object_id``
fields.

:parameter field: the name of the field which will be used to obtained the
    matched elements value. Must be an index.
:rtype: a new :class:`Q` instance.
'''
        if field != self._get_field:
            if field not in self._meta.dfields:
                raise QuerySetError('Model "{0}" has no field "{1}".'
                                    .format(self._meta, field))
            q = self._clone()
            q.data['get_field'] = field
            return q
        else:
            return self
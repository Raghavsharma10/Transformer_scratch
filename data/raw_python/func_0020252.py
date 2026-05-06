def exclude(self, **kwargs):
        '''Returns a new :class:`Query` with additional clauses corresponding
to ``EXCEPT`` in a ``SQL SELECT`` statement.

:parameter kwargs: dictionary of limiting clauses.
:rtype: a new :class:`Query` instance.

Using an equivalent example to the :meth:`filter` method::

    qs = session.query(MyModel)
    result1 = qs.exclude(group = 'planet')
    result2 = qs.exclude(group__in = ('planet','stars'))

'''
        if kwargs:
            q = self._clone()
            if self.eargs:
                kwargs = update_dictionary(self.eargs.copy(), kwargs)
            q.eargs = kwargs
            return q
        else:
            return self
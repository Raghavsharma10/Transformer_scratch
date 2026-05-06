def filter(self, **kwargs):
        '''Create a new :class:`Query` with additional clauses corresponding to
``where`` or ``limit`` in a ``SQL SELECT`` statement.

:parameter kwargs: dictionary of limiting clauses.
:rtype: a new :class:`Query` instance.

For example::

    qs = session.query(MyModel)
    result = qs.filter(group = 'planet')
'''
        if kwargs:
            q = self._clone()
            if self.fargs:
                kwargs = update_dictionary(self.fargs.copy(), kwargs)
            q.fargs = kwargs
            return q
        else:
            return self
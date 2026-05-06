def union(self, *queries):
        '''Return a new :class:`Query` obtained form the union of this
:class:`Query` with one or more *queries*.
For example, lets say we want to have the union
of two queries obtained from the :meth:`filter` method::

    query = session.query(MyModel)
    qs = query.filter(field1 = 'bla').union(query.filter(field2 = 'foo'))
'''
        q = self._clone()
        q.unions += queries
        return q
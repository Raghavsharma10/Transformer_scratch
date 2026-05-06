def intersect(self, *queries):
        '''Return a new :class:`Query` obtained form the intersection of this
:class:`Query` with one or more *queries*. Workds the same way as
the :meth:`union` method.'''
        q = self._clone()
        q.intersections += queries
        return q
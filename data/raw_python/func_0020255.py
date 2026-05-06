def sort_by(self, ordering):
        '''Sort the query by the given field

:parameter ordering: a string indicating the class:`Field` name to sort by.
    If prefixed with ``-``, the sorting will be in descending order, otherwise
    in ascending order.
:return type: a new :class:`Query` instance.
'''
        if ordering:
            ordering = self._meta.get_sorting(ordering, QuerySetError)
        q = self._clone()
        q.data['ordering'] = ordering
        return q
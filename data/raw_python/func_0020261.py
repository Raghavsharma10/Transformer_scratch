def dont_load(self, *fields):
        '''Works like :meth:`load_only` to provides a
:ref:`performance boost <increase-performance>` in cases when you need
to load all fields except a subset specified by *fields*.
'''
        q = self._clone()
        fs = unique_tuple(q.exclude_fields, fields)
        q.exclude_fields = fs if fs else None
        return q
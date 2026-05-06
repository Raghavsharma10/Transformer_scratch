def load_only(self, *fields):
        '''This is provides a :ref:`performance boost <increase-performance>`
in cases when you need to load a subset of fields of your model. The boost
achieved is less than the one obtained when using
:meth:`Query.load_related`, since it does not reduce the number of requests
to the database. However, it can save you lots of bandwidth when excluding
data intensive fields you don't need.
'''
        q = self._clone()
        new_fields = []
        for field in fields:
            if JSPLITTER in field:
                bits = field.split(JSPLITTER)
                related = self._get_related_field(bits[0])
                if related:
                    q._add_to_load_related(related, JSPLITTER.join(bits[1:]))
                    continue
            new_fields.append(field)
        if fields and not new_fields:
            # if we added a field to the load_related list and not fields are
            # are left we add the primary key so that other firls are not
            # loaded.
            new_fields.append(self._meta.pkname())
        fs = unique_tuple(q.fields, new_fields)
        q.data['fields'] = fs if fs else None
        return q
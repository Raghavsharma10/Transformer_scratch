def where(self, code, load_only=None):
        '''For :ref:`backend <db-index>` supporting scripting, it is possible
to construct complex queries which execute the scripting *code* against
each element in the query. The *coe* should reference an instance of
:attr:`model` by ``this`` keyword.

:parameter code: a valid expression in the scripting language of the database.
:parameter load_only: Load only the selected fields when performing the query
    (this is different from the :meth:`load_only` method which is used when
    fetching data from the database). This field is an optimization which is
    used by the :ref:`redis backend <redis-server>` only and can be safely
    ignored in most use-cases.
:return: a new :class:`Query`
'''
        if code:
            q = self._clone()
            q.data['where'] = (code, load_only)
            return q
        else:
            return self
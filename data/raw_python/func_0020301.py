def fieldvalue_pairs(self, exclude_cache=False):
        '''Generator of fields,values pairs. Fields correspond to
the ones which have been loaded (usually all of them) or
not loaded but modified.
Check the :ref:`load_only <performance-loadonly>` query function for more
details.

If *exclude_cache* evaluates to ``True``, fields with :attr:`Field.as_cache`
attribute set to ``True`` won't be included.

:rtype: a generator of two-elements tuples'''
        for field in self._meta.scalarfields:
            if exclude_cache and field.as_cache:
                continue
            name = field.attname
            if hasattr(self, name):
                yield field, getattr(self, name)
def clear_cache_fields(self):
        '''Set cache fields to ``None``. Check :attr:`Field.as_cache`
for information regarding fields which are considered cache.'''
        for field in self._meta.scalarfields:
            if field.as_cache:
                setattr(self, field.name, None)
def todict(self, exclude_cache=False):
        '''Return a dictionary of serialised scalar field for pickling.
If the *exclude_cache* flag is ``True``, fields with :attr:`Field.as_cache`
attribute set to ``True`` will be excluded.'''
        odict = {}
        for field, value in self.fieldvalue_pairs(exclude_cache=exclude_cache):
            value = field.serialise(value)
            if value:
                odict[field.name] = value
        if self._dbdata and 'id' in self._dbdata:
            odict['__dbdata__'] = {'id': self._dbdata['id']}
        return odict
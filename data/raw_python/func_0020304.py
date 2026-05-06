def clone(self, **data):
        '''Utility method for cloning the instance as a new object.

:parameter data: additional which override field data.
:rtype: a new instance of this class.
'''
        meta = self._meta
        session = self.session
        pkname = meta.pkname()
        pkvalue = data.pop(pkname, None)
        fields = self.todict(exclude_cache=True)
        fields.update(data)
        fields.pop('__dbdata__', None)
        obj = self._meta.make_object((pkvalue, None, fields))
        obj.session = session
        return obj
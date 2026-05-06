def get_state(self, **kwargs):
        '''Return the current :class:`ModelState` for this :class:`Model`.
If ``kwargs`` parameters are passed a new :class:`ModelState` is created,
otherwise it returns the cached value.'''
        dbdata = self.dbdata
        if 'state' not in dbdata or kwargs:
            dbdata['state'] = ModelState(self, **kwargs)
        return dbdata['state']
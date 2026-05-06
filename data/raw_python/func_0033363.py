def get_full_history(self, force=None, last_update=None, flush=False):
        '''
        Fields change depending on when you run activity_import,
        such as "last_updated" type fields which don't have activity
        being tracked, which means we'll always end up with different
        hash values, so we need to always remove all existing object
        states and import fresh
        '''
        return self._run_object_import(force=force, last_update=last_update,
                                       flush=flush, full_history=True)
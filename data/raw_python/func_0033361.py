def get_objects(self, force=None, last_update=None, flush=False):
        '''
        Extract routine for SQL based cubes.

        :param force:
            for querying for all objects (True) or only those passed in as list
        :param last_update: manual override for 'changed since date'
        '''
        return self._run_object_import(force=force, last_update=last_update,
                                       flush=flush, full_history=False)
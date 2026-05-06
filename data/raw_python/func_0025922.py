def _get_changes(self):
        '''Get all changed values.'''
        result = dict( (f['id'], f.get('value','')) for f in self._data if f.get('changed', False) )
        self._clear_changes
        return result
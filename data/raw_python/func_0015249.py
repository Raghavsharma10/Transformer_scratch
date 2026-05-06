def assistants(self):
        '''Get all assistants in this DAP'''
        return [strip_suffix(f, '.yaml') for f in self._stripped_files if self._assistants_pattern.match(f)]
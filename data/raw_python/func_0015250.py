def snippets(self):
        '''Get all snippets in this DAP'''
        return [strip_suffix(f, '.yaml') for f in self._stripped_files if self._snippets_pattern.match(f)]
def icons(self, strip_ext=False):
        '''Get all icons in this DAP, optionally strip extensions'''
        result =  [f for f in self._stripped_files if self._icons_pattern.match(f)]
        if strip_ext:
            result = [strip_suffix(f, '\.({ext})'.format(ext=self._icons_ext), regex=True) for f in result]

        return result
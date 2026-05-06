def _asStr(self):
        '''
            _asStr - Get the string representation of this style

              @return <str> - A string representation of this style (semicolon separated, key: value format)
        '''

        styleDict = self._styleDict
        if styleDict:
            return '; '.join([name + ': ' + value for name, value in styleDict.items()])
        return ''
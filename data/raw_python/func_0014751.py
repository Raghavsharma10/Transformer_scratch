def get(self, key, default=None):
        '''
            get - Gets an attribute by key with the chance to provide a default value

                @param key <str> - The key to query

                @param default <Anything> Default None - The value to return if key is not found

             @return - The value of attribute at #key, or #default if not present.
        '''

        key = key.lower()

        if key == 'class':
            return self.tag.className

        if key in ('style', 'class') or key in self.keys():
            return self[key]
        return default
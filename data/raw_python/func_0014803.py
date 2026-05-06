def get_all_data_from_cache(self, filename=''):
        ''' Reads the JSON inventory from the cache file. Returns Python dictionary. '''

        data = ''
        if not filename:
            filename = self.cache_path_cache
        with open(filename, 'r') as cache:
            data = cache.read()
        return json.loads(data)
def write_to_cache(self, data, filename=''):
        ''' Writes data to file as JSON.  Returns True. '''
        if not filename:
            filename = self.cache_path_cache
        json_data = json.dumps(data)
        with open(filename, 'w') as cache:
            cache.write(json_data)
        return True
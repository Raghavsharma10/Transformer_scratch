def write_to_cache(self, data, filename):
        ''' Writes data in JSON format to a file '''

        json_data = json.dumps(data, sort_keys=True, indent=2)
        cache = open(filename, 'w')
        cache.write(json_data)
        cache.close()
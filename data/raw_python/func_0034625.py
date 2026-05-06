def load_inventory_from_cache(self):
        ''' Reads the inventory from the cache file and returns it as a JSON
        object '''

        cache = open(self.cache_path_cache, 'r')
        json_inventory = cache.read()
        self.inventory = json.loads(json_inventory)
def load_inventory_from_cache(self):
        ''' Loads inventory from JSON on disk. '''

        try:
            self.inventory = self.cache.get_all_data_from_cache()
            hosts = self.inventory['_meta']['hostvars']
        except Exception as e:
            print(
                "Invalid inventory file %s.  Please rebuild with -refresh-cache option."
                % (self.cache.cache_path_cache))
            raise
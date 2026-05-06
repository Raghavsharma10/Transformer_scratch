def do_api_calls_update_cache(self):
        ''' Do API calls and save data in cache. '''
        zones = self.parse_env_zones()
        data = self.group_instances(zones)
        self.cache.write_to_cache(data)
        self.inventory = data
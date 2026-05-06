def get_host(self, host):
        ''' Get variables about a specific host '''

        if len(self.index) == 0:
            # Need to load index from cache
            self.load_index_from_cache()

        if not host in self.index:
            # try updating the cache
            self.do_api_calls_update_cache()
            if not host in self.index:
                # host might not exist anymore
                return {}

        (region, instance_id) = self.index[host]

        instance = self.get_instance(region, instance_id)
        return self.get_host_info_dict_from_instance(instance)
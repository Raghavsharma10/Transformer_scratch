def is_valid(self, max_age=None):
        ''' Determines if the cache files have expired, or if it is still valid '''

        if max_age is None:
            max_age = self.cache_max_age

        if os.path.isfile(self.cache_path_cache):
            mod_time = os.path.getmtime(self.cache_path_cache)
            current_time = time()
            if (mod_time + max_age) > current_time:
                return True

        return False
def cache_persist(self):
        """
        Saves the current trained data to the cache.
        This is initiated by the program using this module
        """
        filename = self.get_cache_location()
        pickle.dump(self.categories, open(filename, 'wb'))
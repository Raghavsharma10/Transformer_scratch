def get_cache_location(self):
        """
        Gets the location of the cache file

        :return: the location of the cache file
        :rtype: string
        """
        filename = self.cache_path if \
            self.cache_path[-1:] == '/' else \
            self.cache_path + '/'
        filename += self.cache_file
        return filename
def lookup_name_slow(self, name):
        """Returns a struct if one exists"""

        for index in xrange(self.__get_count_cached()):
            if self.__get_name_cached(index) == name:
                return self.__get_info_cached(index)
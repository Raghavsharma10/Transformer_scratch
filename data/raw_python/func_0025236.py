def __initialize_cache(self):
        """Initialize the cache values (cache values are used for optimization)."""
        if self.__cached_value_dirty is None:
            self.__cached_value_dirty = self.__cache.is_cached_value_dirty(self.__display_item, self.__cache_property_name)
            self.__cached_value = self.__cache.get_cached_value(self.__display_item, self.__cache_property_name)
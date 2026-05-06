def mark_data_dirty(self):
        """ Called from item to indicate its data or metadata has changed."""
        self.__cache.set_cached_value_dirty(self.__display_item, self.__cache_property_name)
        self.__initialize_cache()
        self.__cached_value_dirty = True
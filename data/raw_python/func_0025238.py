def recompute_data(self, ui):
        """Compute the data associated with this processor.

        This method is thread safe and may take a long time to return. It should not be called from
         the UI thread. Upon return, the results will be calculated with the latest data available
         and the cache will not be marked dirty.
        """
        self.__initialize_cache()
        with self.__recompute_lock:
            if self.__cached_value_dirty:
                try:
                    calculated_data = self.get_calculated_data(ui)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    traceback.print_stack()
                    raise
                self.__cache.set_cached_value(self.__display_item, self.__cache_property_name, calculated_data)
                self.__cached_value = calculated_data
                self.__cached_value_dirty = False
                self.__cached_value_time = time.time()
            else:
                calculated_data = None
            if calculated_data is None:
                calculated_data = self.get_default_data()
                if calculated_data is not None:
                    # if the default is not None, treat is as valid cached data
                    self.__cache.set_cached_value(self.__display_item, self.__cache_property_name, calculated_data)
                    self.__cached_value = calculated_data
                    self.__cached_value_dirty = False
                    self.__cached_value_time = time.time()
                else:
                    # otherwise remove everything from the cache
                    self.__cache.remove_cached_value(self.__display_item, self.__cache_property_name)
                    self.__cached_value = None
                    self.__cached_value_dirty = None
                    self.__cached_value_time = 0
            self.__recompute_lock.release()
            if callable(self.on_thumbnail_updated):
                self.on_thumbnail_updated()
            self.__recompute_lock.acquire()
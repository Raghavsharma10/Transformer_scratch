def decrement_display_ref_count(self, amount: int=1):
        """Decrement display reference count to indicate this library item is no longer displayed."""
        assert not self._closed
        self.__display_ref_count -= amount
        if self.__display_ref_count == 0:
            self.__is_master = False
        if self.__data_item:
            for _ in range(amount):
                self.__data_item.decrement_data_ref_count()
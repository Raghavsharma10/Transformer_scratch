def increment_display_ref_count(self, amount: int=1):
        """Increment display reference count to indicate this library item is currently displayed."""
        display_ref_count = self.__display_ref_count
        self.__display_ref_count += amount
        if display_ref_count == 0:
            self.__is_master = True
        if self.__data_item:
            for _ in range(amount):
                self.__data_item.increment_data_ref_count()
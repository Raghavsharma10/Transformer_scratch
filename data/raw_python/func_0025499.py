def increment_display_ref_count(self, amount: int=1):
        """Increment display reference count to indicate this library item is currently displayed."""
        display_ref_count = self.__display_ref_count
        self.__display_ref_count += amount
        for display_data_channel in self.display_data_channels:
            display_data_channel.increment_display_ref_count(amount)
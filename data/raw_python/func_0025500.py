def decrement_display_ref_count(self, amount: int=1):
        """Decrement display reference count to indicate this library item is no longer displayed."""
        assert not self._closed
        self.__display_ref_count -= amount
        for display_data_channel in self.display_data_channels:
            display_data_channel.decrement_display_ref_count(amount)
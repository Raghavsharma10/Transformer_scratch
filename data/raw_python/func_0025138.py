def start(self):
        """Called from hardware source when data starts streaming."""
        old_start_count = self.__start_count
        self.__start_count += 1
        if old_start_count == 0:
            self.data_channel_start_event.fire()
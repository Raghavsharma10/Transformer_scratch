def close(self) -> None:
        """Close the task.

        .. versionadded:: 1.0

        This method must be called when the task is no longer needed.
        """
        self.__data_channel_buffer.stop()
        self.__data_channel_buffer.close()
        self.__data_channel_buffer = None
        if not self.__was_playing:
            self.__hardware_source.stop_playing()
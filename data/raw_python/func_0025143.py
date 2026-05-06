def pause(self) -> None:
        """Pause recording.

        Thread safe and UI safe."""
        with self.__state_lock:
            if self.__state == DataChannelBuffer.State.started:
                self.__state = DataChannelBuffer.State.paused
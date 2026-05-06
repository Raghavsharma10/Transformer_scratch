def resume(self) -> None:
        """Resume recording after pause.

        Thread safe and UI safe."""
        with self.__state_lock:
            if self.__state == DataChannelBuffer.State.paused:
                self.__state = DataChannelBuffer.State.started
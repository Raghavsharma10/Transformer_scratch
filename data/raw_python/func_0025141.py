def grab_next(self, timeout: float=None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        """Grab the next data to finish from the buffer, blocking until one is available."""
        with self.__buffer_lock:
            self.__buffer = list()
        return self.grab_latest(timeout)
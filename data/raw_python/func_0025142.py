def grab_following(self, timeout: float=None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        """Grab the next data to start from the buffer, blocking until one is available."""
        self.grab_next(timeout)
        return self.grab_next(timeout)
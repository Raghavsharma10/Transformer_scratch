def write(self, data: bytes) -> None:
        """
        Write the data.
        """
        if self.finished():
            if self._exc:
                raise self._exc

            raise WriteAfterFinishedError

        if not data:
            return

        try:
            self._delegate.write_data(data, finished=False)

        except BaseWriteException as e:
            self._finished.set()
            if self._exc is None:
                self._exc = e

            raise
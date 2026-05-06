def finish(self, data: bytes=b"") -> None:
        """
        Finish the stream.
        """
        if self.finished():
            if self._exc:
                raise self._exc

            if data:
                raise WriteAfterFinishedError

            return

        try:
            self._delegate.write_data(data, finished=True)

        except BaseWriteException as e:
            if self._exc is None:
                self._exc = e

            raise

        finally:
            self._finished.set()
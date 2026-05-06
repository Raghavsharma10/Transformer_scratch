async def read(self, n: int=-1, exactly: bool=False) -> bytes:
        """
        Read at most n bytes data or if exactly is `True`,
        read exactly n bytes data. If the end has been reached before
        the buffer has the length of data asked, it will
        raise a :class:`ReadUnsatisfiableError`.

        When :method:`.finished()` is `True`, this method will raise any errors
        occurred during the read or a :class:`ReadFinishedError`.
        """
        async with self._read_lock:
            self._raise_exc_if_finished()

            if n == 0:
                return b""

            if exactly:
                if n < 0:  # pragma: no cover
                    raise ValueError(
                        "You MUST sepcify the length of the data "
                        "if exactly is True.")

                if n > self.max_buf_len:  # pragma: no cover
                    raise ValueError(
                        "The length provided cannot be larger "
                        "than the max buffer length.")

                while len(self) < n:
                    try:
                        await self._wait_for_data()

                    except asyncio.CancelledError:  # pragma: no cover
                        raise

                    except Exception as e:
                        raise ReadUnsatisfiableError from e

            elif n < 0:
                while True:
                    if len(self) > self.max_buf_len:
                        raise MaxBufferLengthReachedError

                    try:
                        await self._wait_for_data()

                    except asyncio.CancelledError:  # pragma: no cover
                        raise

                    except Exception:
                        data = bytes(self._buf)
                        self._buf.clear()

                        return data

            elif len(self) == 0:
                await self._wait_for_data()

            data = bytes(self._buf[0:n])
            del self._buf[0:n]

            return data
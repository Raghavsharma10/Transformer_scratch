async def read_until(
        self, separator: bytes=b"\n",
            *, keep_separator: bool=True) -> bytes:
        """
        Read until the separator has been found.

        When the max size of the buffer has been reached,
        and the separator is not found, this method will raise
        a :class:`MaxBufferLengthReachedError`.
        Similarly, if the end has been reached before found the separator
        it will raise a :class:`SeparatorNotFoundError`.

        When :method:`.finished()` is `True`, this method will raise any errors
        occurred during the read or a :class:`ReadFinishedError`.
        """
        async with self._read_lock:
            self._raise_exc_if_finished()

            start_pos = 0

            while True:
                separator_pos = self._buf.find(separator, start_pos)

                if separator_pos != -1:
                    break

                if len(self) > self.max_buf_len:
                    raise MaxBufferLengthReachedError

                try:
                    await self._wait_for_data()

                except asyncio.CancelledError:  # pragma: no cover
                    raise

                except Exception as e:
                    if len(self) > 0:
                        raise SeparatorNotFoundError from e

                    else:
                        raise

                new_start_pos = len(self) - len(separator)

                if new_start_pos > 0:
                    start_pos = new_start_pos

            full_pos = separator_pos + len(separator)

            if keep_separator:
                data_pos = full_pos

            else:
                data_pos = separator_pos

            data = bytes(self._buf[0:data_pos])
            del self._buf[0:full_pos]

            return data
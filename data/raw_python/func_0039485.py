async def flush(self) -> None:
        """
        Give the writer a chance to flush the pending data
        out of the internal buffer.
        """
        async with self._flush_lock:
            if self.finished():
                if self._exc:
                    raise self._exc

                return

            try:
                await self._delegate.flush_buf()

            except asyncio.CancelledError:  # pragma: no cover
                raise

            except BaseWriteException as e:
                self._finished.set()
                if self._exc is None:
                    self._exc = e

                raise
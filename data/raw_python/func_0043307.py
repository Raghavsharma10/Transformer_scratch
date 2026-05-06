def close(self):
        """
        Close the instance.
        """
        if not self.closed and not self.closing:
            logger.debug(
                "%s[%s] closing...",
                self.__class__.__name__,
                id(self),
            )
            self._closing.set()
            future = asyncio.ensure_future(self.on_close(), loop=self.loop)
            future.add_done_callback(self._set_closed)
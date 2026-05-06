def _set_closed(self, future):
        """
        Indicate that the instance is effectively closed.

        :param future: The close future.
        """
        logger.debug("%s[%s] closed.", self.__class__.__name__, id(self))
        self.on_closed.emit(self)
        self._closed_future.set_result(future.result())
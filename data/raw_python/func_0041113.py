def callback(self, herald_svc, message):
        """
        Tries to call the callback of the post message.
        Avoids errors to go outside this method.

        :param herald_svc: Herald service instance
        :param message: Received answer message
        """
        if self.__callback is not None:
            try:
                # pylint: disable=W0703
                self.__callback(herald_svc, message)
            except Exception as ex:
                _logger.exception("Error calling callback: %s", ex)
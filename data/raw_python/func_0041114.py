def errback(self, herald_svc, exception):
        """
        Tries to call the error callback of the post message.
        Avoids errors to go outside this method.

        :param herald_svc: Herald service instance
        :param exception: An exception describing/caused by the error
        """
        if self.__errback is not None:
            try:
                # pylint: disable=W0703
                self.__errback(herald_svc, exception)
            except Exception as ex:
                _logger.exception("Error calling errback: %s", ex)
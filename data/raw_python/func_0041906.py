def __callback(self, data):
        """
        Safely calls back a method

        :param data: Associated stanza
        """
        method = self.__cb_message
        if method is not None:
            try:
                method(data)
            except Exception as ex:
                _logger.exception("Error calling method: %s", ex)
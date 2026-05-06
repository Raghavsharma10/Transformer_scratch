def _raise_unrecoverable_error_client(self, exception):
        """
        Raises an exceptions.ClientError with a message telling that the error probably comes from the client
        configuration.
        :param exception: Exception that caused the ClientError
        :type exception: Exception
        :raise exceptions.ClientError
        """
        message = ('There was an unrecoverable error during the HTTP request which is probably related to your '
                   'configuration. Please verify `' + self.DEPENDENCY + '` library configuration and update it. If the '
                   'issue persists, do not hesitate to contact us with the following information: `' + repr(exception) +
                   '`.')
        raise exceptions.ClientError(message, client_exception=exception)
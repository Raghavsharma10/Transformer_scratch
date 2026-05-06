def _raise_unrecoverable_error_payplug(self, exception):
        """
        Raises an exceptions.ClientError with a message telling that the error probably comes from PayPlug.
        :param exception: Exception that caused the ClientError.
        :type exception: Exception
        :raise exceptions.ClientError
        """
        message = ('There was an unrecoverable error during the HTTP request. It seems to come from our servers. '
                   'If you are behind a proxy, ensure that it is configured correctly. If the issue persists, do not '
                   'hesitate to contact us with the following information: `' + repr(exception) + '`.')
        raise exceptions.ClientError(message, client_exception=exception)
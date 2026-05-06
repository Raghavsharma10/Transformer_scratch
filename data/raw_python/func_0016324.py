def raise_for_response(self, responses):
        """
        Constructs appropriate exception from list of responses and raises it.
        """
        exception_messages = [self.client.format_exception_message(response) for response in responses]
        if len(exception_messages) == 1:
            message = exception_messages[0]
        else:
            message = "[%s]" % ", ".join(exception_messages)
        raise PostmarkerException(message)
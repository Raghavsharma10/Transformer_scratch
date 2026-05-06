def process(self):
        """ Receive a subscription from the socket and process it """

        subscription = None
        result = None

        try:
            subscription = self.socket.recv()

        except AuthenticateError as exception:
            logging.error(
                'Subscriber error while authenticating request: {}'
                .format(exception), exc_info=1)

        except AuthenticatorInvalidSignature as exception:
            logging.error(
                'Subscriber error while authenticating request: {}'
                .format(exception), exc_info=1)

        except DecodeError as exception:
            logging.error(
                'Subscriber error while decoding request: {}'
                .format(exception), exc_info=1)

        except RequestParseError as exception:
            logging.error(
                'Subscriber error while parsing request: {}'
                .format(exception), exc_info=1)

        else:
            logging.debug(
                'Subscriber received payload: {}'
                .format(subscription))

        _tag, message, fun = self.parse(subscription)
        message = self.verify(message)
        message = self.decode(message)

        try:
            result = fun(message)
        except Exception as exception:
            logging.error(exception, exc_info=1)

        # Return result to check successful execution of `fun` when testing
        return result
def process(self):
        """ Receive data from socket and process request """

        response = None

        try:
            payload = self.receive()
            method, args, ref = self.parse(payload)
            response = self.execute(method, args, ref)

        except AuthenticateError as exception:
            logging.error(
                'Service error while authenticating request: {}'
                .format(exception), exc_info=1)

        except AuthenticatorInvalidSignature as exception:
            logging.error(
                'Service error while authenticating request: {}'
                .format(exception), exc_info=1)

        except DecodeError as exception:
            logging.error(
                'Service error while decoding request: {}'
                .format(exception), exc_info=1)

        except RequestParseError as exception:
            logging.error(
                'Service error while parsing request: {}'
                .format(exception), exc_info=1)

        else:
            logging.debug('Service received payload: {}'.format(payload))

        if response:
            self.send(response)
        else:
            self.send('')
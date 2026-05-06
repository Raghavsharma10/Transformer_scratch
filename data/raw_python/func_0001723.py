def call(self, method, *args):
        """ Make a call to a `Responder` and return the result """

        payload = self.build_payload(method, args)
        logging.debug('* Client will send payload: {}'.format(payload))
        self.send(payload)

        res = self.receive()
        assert payload[2] == res['ref']
        return res['result'], res['error']
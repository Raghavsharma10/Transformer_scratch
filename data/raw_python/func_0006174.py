def wait_for_responses(self, client):
        """Waits for all responses to come back and resolves the
        eventual results.
        """
        assert_open(self)

        if self.has_pending_requests:
            raise RuntimeError('Cannot wait for responses if there are '
                               'pending requests outstanding.  You need '
                               'to wait for pending requests to be sent '
                               'first.')

        pending = self.pending_responses
        self.pending_responses = []
        for command_name, options, promise in pending:
            value = client.parse_response(
                self.connection, command_name, **options)
            promise.resolve(value)
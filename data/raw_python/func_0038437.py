def _get_err(self, e, id=None, jsonrpc=DEFAULT_JSONRPC):
        """
        Returns jsonrpc error message.
        """
        # Do not respond to notifications when the request is valid.
        if not id \
                and not isinstance(e, ParseError) \
                and not isinstance(e, InvalidRequestError):
            return None

        respond = {'id': id}

        if isinstance(jsonrpc, int):
            # v1.0 requires result to exist always.
            # No error codes are defined in v1.0 so only use the message.
            if jsonrpc == 10:
                respond['result'] = None
                respond['error'] = e.dumps()['message']
            else:
                self._fill_ver(jsonrpc, respond)
                respond['error'] = e.dumps()
        else:
            respond['jsonrpc'] = jsonrpc
            respond['error'] = e.dumps()

        return respond
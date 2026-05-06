def _fill_request(self, request, rdata):
        """Fills request with data from the jsonrpc call."""
        if not isinstance(rdata, dict):
            raise InvalidRequestError

        request['jsonrpc'] = self._get_jsonrpc(rdata)
        request['id'] = self._get_id(rdata)
        request['method'] = self._get_method(rdata)
        request['params'] = self._get_params(rdata)
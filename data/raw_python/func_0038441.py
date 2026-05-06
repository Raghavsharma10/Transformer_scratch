def _get_method(self, rdata):
        """
        Returns jsonrpc request's method value.

        InvalidRequestError will be raised if it's missing or is wrong type.
        MethodNotFoundError will be raised if a method with given method name
        does not exist.
        """
        if 'method' in rdata:
            if not isinstance(rdata['method'], basestring):
                raise InvalidRequestError
        else:
            raise InvalidRequestError

        if rdata['method'] not in self.method_data.keys():
            raise MethodNotFoundError

        return rdata['method']
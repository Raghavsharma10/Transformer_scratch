def _call_method(self, request):
        """Calls given method with given params and returns it value."""
        method = self.method_data[request['method']]['method']
        params = request['params']
        result = None
        try:
            if isinstance(params, list):
                # Does it have enough arguments?
                if len(params) < self._man_args(method):
                    raise InvalidParamsError('not enough arguments')
                # Does it have too many arguments?
                if not self._vargs(method) \
                        and len(params) > self._max_args(method):
                    raise InvalidParamsError('too many arguments')

                result = yield defer.maybeDeferred(method, *params)
            elif isinstance(params, dict):
                # Do not accept keyword arguments if the jsonrpc version is
                # not >=1.1.
                if request['jsonrpc'] < 11:
                    raise KeywordError

                result = yield defer.maybeDeferred(method, **params)
            else:  # No params
                result = yield defer.maybeDeferred(method)
        except JSONRPCError:
            raise
        except Exception:
            # Exception was raised inside the method.
            log.msg('Exception raised while invoking RPC method "{}".'.format(
                    request['method']))
            log.err()
            raise ServerError

        defer.returnValue(result)
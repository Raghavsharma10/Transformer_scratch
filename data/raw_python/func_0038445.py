def _handle_request(self, request):
        """Handles given request and returns its response."""
        if 'types' in self.method_data[request['method']]:
            self._validate_params_types(request['method'], request['params'])

        if self.serve_exception:
            raise self.serve_exception()
        d = self._call_method(request)
        self.pending.add(d)
        if self.timeout:
            timeout_deferred = self.reactor.callLater(self.timeout, d.cancel)

            def completed(result):
                if timeout_deferred.active():
                    # cancel the timeout_deferred if it has not been fired yet
                    # this is to prevent d's deferred chain from firing twice
                    # (and raising an exception).
                    timeout_deferred.cancel()
                return result
            d.addBoth(completed)
        try:
            result = yield d
        except defer.CancelledError:
            # The request was cancelled due to a timeout or by cancelPending
            # having been called. We return a TimeoutError to the client.
            self._remove_pending(d)
            raise TimeoutError()
        except Exception as e:
            self._remove_pending(d)
            raise e
        self._remove_pending(d)
        # Do not respond to notifications.
        if request['id'] is None:
            defer.returnValue(None)

        respond = {}
        self._fill_ver(request['jsonrpc'], respond)
        respond['result'] = result
        respond['id'] = request['id']

        defer.returnValue(respond)
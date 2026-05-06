def _register_server(self, server, timeout=30):
        '''Register a new SiriDB Server.

        This method is used by the SiriDB manage tool and should not be used
        otherwise. Full access rights are required for this request.
        '''
        result = self._loop.run_until_complete(
            self._protocol.send_package(CPROTO_REQ_REGISTER_SERVER,
                                        data=server,
                                        timeout=timeout))
        return result
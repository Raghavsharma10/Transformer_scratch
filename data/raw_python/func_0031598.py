def rpc(self, cmd, **kwargs):
        """Generic helper function to call an RPC method."""

        func = getattr(self.client, cmd)
        try:
            if self.credentials is None:
                return func(kwargs)
            else:
                return func(self.credentials, kwargs)
        except socket.error as e:
            raise BackendConnectionError(e)
        except (xmlrpclib.ProtocolError, BadStatusLine) as e:
            log.error(e)
            raise BackendError("Error reaching backend.")
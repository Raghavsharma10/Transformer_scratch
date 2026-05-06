def startService(self):
        """
        Start the service and connect the JSONRPCClientFactory.
        """
        self.clientFactory.connect().addErrback(
            log.err, 'error starting the JSON-RPC client service %r' % (self,))
        service.Service.startService(self)
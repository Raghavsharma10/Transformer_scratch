def stopServing(self, exception=None):
        """
        Returns a deferred that will fire immediately if there are
        no pending requests, otherwise when the last request is removed
        from self.pending.
        """
        if exception is None:
            exception = ServiceUnavailableError
        self.serve_exception = exception
        if self.pending:
            d = self.out_of_service_deferred = defer.Deferred()
            return d
        return defer.succeed(None)
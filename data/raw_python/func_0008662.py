def killall(self, exc):
        """ Connection/Channel was closed. All subsequent and ongoing requests
            should raise an error
        """
        self.connection_exc = exc
        # Set an exception for all others
        for method, futs in self._futures.items():
            for fut in futs:
                if fut.done():
                    continue
                fut.set_exception(exc)
        self._futures.clear()
def listen_until_return(self, *temporary_handlers, timeout=0):
        """Calls listen repeatedly until listen returns something else than None.
        Then returns listen's result. If timeout is not zero listen_until_return
        stops after timeout seconds and returns None."""
        start = time.time()
        while timeout == 0 or time.time() - start < timeout:
            res = self.listen(*temporary_handlers)
            if res is not None:
                return res
def wait_until_exit(self):
        """ Wait until thread exit

            Used for testing purpose only
        """

        if self._timeout is None:
            raise Exception("Thread will never exit. Use stop or specify timeout when starting it!")

        self._thread.join()
        self.stop()
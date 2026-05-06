def wait(self, timeout=None):
        """
        Waits for the client to stop its loop
        """
        self.__stopped.wait(timeout)
        return self.__stopped.is_set()
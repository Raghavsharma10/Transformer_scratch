def close(self):
        """
        Close all endpoint file descriptors.
        """
        ep_list = self._ep_list
        while ep_list:
            ep_list.pop().close()
        self._closed = True
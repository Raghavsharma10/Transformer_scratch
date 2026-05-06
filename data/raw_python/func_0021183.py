def flush_and_refresh(self, index):
        """Flush and refresh one or more indices.

        .. warning::

           Do not call this method unless you know what you are doing. This
           method is only intended to be called during tests.
        """
        self.client.indices.flush(wait_if_ongoing=True, index=index)
        self.client.indices.refresh(index=index)
        self.client.cluster.health(
            wait_for_status='yellow', request_timeout=30)
        return True
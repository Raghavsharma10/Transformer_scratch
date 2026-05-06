def start(self, timeout=120):
        """
        Start the server. Note: slow and blocking request.

        The API waits for confirmation from UpCloud's IaaS backend before responding.
        """
        path = '/server/{0}/start'.format(self.uuid)
        self.cloud_manager.post_request(path, timeout=timeout)
        object.__setattr__(self, 'state', 'started')
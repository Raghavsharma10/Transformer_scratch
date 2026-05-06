def get_handler(self):
        """Create a fully configured CloudLoggingHandler.

        Returns:
            (obj): Instance of `google.cloud.logging.handlers.
                                CloudLoggingHandler`
        """

        gcl_client = gcl_logging.Client(
            project=self.project_id, credentials=self.credentials)
        handler = gcl_handlers.CloudLoggingHandler(
            gcl_client,
            resource=self.resource,
            labels={
                'resource_id': self.instance_id,
                'resource_project': self.project_id,
                'resource_zone': self.zone,
                'resource_host': self.hostname
            })
        handler.setFormatter(self.get_formatter())
        self._set_worker_thread_level()
        return handler
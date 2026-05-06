def _create_gcl_resource(self):
        """Create a configured Resource object.

        The logging.resource.Resource object enables GCL to filter and
        bucket incoming logs according to which resource (host) they're
        coming from.

        Returns:
            (obj): Instance of `google.cloud.logging.resource.Resource`
        """

        return gcl_resource.Resource('gce_instance', {
            'project_id': self.project_id,
            'instance_id': self.instance_id,
            'zone': self.zone
        })
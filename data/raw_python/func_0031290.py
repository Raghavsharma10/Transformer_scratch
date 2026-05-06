def wait_for_instance_deletion(self, credentials, name, **kwargs):
        """Wait for deletion of instance based on the configuration data.

        TODO: docstring"""
        op_name = wait_for_instance_deletion(
            credentials, self.project, self.zone, name, **kwargs)

        return op_name
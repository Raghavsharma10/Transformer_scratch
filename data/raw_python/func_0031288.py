def create_instance(self, credentials, name, **kwargs):
        """Create an instance based on the configuration data.
        
        TODO: docstring"""
        op_name = create_instance(
            credentials, self.project, self.zone, name,
            machine_type=self.machine_type,
            disk_size_gb=self.disk_size_gb,
            **kwargs)
        
        return op_name
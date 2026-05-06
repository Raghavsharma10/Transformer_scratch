def create(self):
        """Create the required directory structure and admin metadata."""
        self._storage_broker.create_structure()
        self._storage_broker.put_admin_metadata(self._admin_metadata)
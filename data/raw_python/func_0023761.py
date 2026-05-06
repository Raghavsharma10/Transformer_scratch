def add_storage(self, storage=None, type='disk', address=None):
        """
        Attach the given storage to the Server.

        Default address is next available.
        """
        self.cloud_manager.attach_storage(server=self.uuid,
                                          storage=storage.uuid,
                                          storage_type=type,
                                          address=address)
        storage.address = address
        storage.type = type
        self.storage_devices.append(storage)
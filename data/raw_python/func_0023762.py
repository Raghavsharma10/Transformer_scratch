def remove_storage(self, storage):
        """
        Remove Storage from a Server.

        The Storage must be a reference to an object in
        Server.storage_devices or the method will throw and Exception.

        A Storage from get_storage(uuid) will not work as it is missing the 'address' property.
        """
        if not hasattr(storage, 'address'):
            raise Exception(
                ('Storage does not have an address. '
                 'Access the Storage via Server.storage_devices '
                 'so they include an address. '
                 '(This is due how the API handles Storages)')
            )

        self.cloud_manager.detach_storage(server=self.uuid, address=storage.address)
        self.storage_devices.remove(storage)
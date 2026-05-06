def populate(self):
        """
        Sync changes from the API to the local object.

        Note: syncs ip_addresses and storage_devices too (/server/uuid endpoint)
        """
        server, IPAddresses, storages = self.cloud_manager.get_server_data(self.uuid)
        self._reset(
            server,
            ip_addresses=IPAddresses,
            storage_devices=storages,
            populated=True
        )
        return self
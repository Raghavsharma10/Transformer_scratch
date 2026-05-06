def get_server(self, UUID):
        """
        Return a (populated) Server instance.
        """
        server, IPAddresses, storages = self.get_server_data(UUID)

        return Server(
            server,
            ip_addresses=IPAddresses,
            storage_devices=storages,
            populated=True,
            cloud_manager=self
        )
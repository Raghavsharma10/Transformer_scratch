def get_server_data(self, UUID):
        """
        Return '/server/uuid' data in Python dict.

        Creates object representations of any IP-address and Storage.
        """
        data = self.get_request('/server/{0}'.format(UUID))
        server = data['server']

        # Populate subobjects
        IPAddresses = IPAddress._create_ip_address_objs(server.pop('ip_addresses'),
                                                          cloud_manager=self)

        storages = Storage._create_storage_objs(server.pop('storage_devices'),
                                                cloud_manager=self)

        return server, IPAddresses, storages
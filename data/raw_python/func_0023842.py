def modify_server(self, UUID, **kwargs):
        """
        modify_server allows updating the server's updateable_fields.

        Note: Server's IP-addresses and Storages are managed by their own add/remove methods.
        """
        body = dict()
        body['server'] = {}
        for arg in kwargs:
            if arg not in Server.updateable_fields:
                Exception('{0} is not an updateable field'.format(arg))
            body['server'][arg] = kwargs[arg]

        res = self.request('PUT', '/server/{0}'.format(UUID), body)
        server = res['server']

        # Populate subobjects
        IPAddresses = IPAddress._create_ip_address_objs(server.pop('ip_addresses'),
                                                          cloud_manager=self)

        storages = Storage._create_storage_objs(server.pop('storage_devices'),
                                                cloud_manager=self)

        return Server(
            server,
            ip_addresses=IPAddresses,
            storage_devices=storages,
            populated=True,
            cloud_manager=self
        )
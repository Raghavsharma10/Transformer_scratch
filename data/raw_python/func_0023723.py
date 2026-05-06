def detach_storage(self, server, address):
        """
        Detach a Storage object to a Server. Return a list of the server's storages.
        """
        body = {'storage_device': {'address': address}}
        url = '/server/{0}/storage/detach'.format(server)
        res = self.post_request(url, body)
        return Storage._create_storage_objs(res['server']['storage_devices'], cloud_manager=self)
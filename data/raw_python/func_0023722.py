def attach_storage(self, server, storage, storage_type, address):
        """
        Attach a Storage object to a Server. Return a list of the server's storages.
        """
        body = {'storage_device': {}}
        if storage:
            body['storage_device']['storage'] = str(storage)

        if storage_type:
            body['storage_device']['type'] = storage_type

        if address:
            body['storage_device']['address'] = address

        url = '/server/{0}/storage/attach'.format(server)
        res = self.post_request(url, body)
        return Storage._create_storage_objs(res['server']['storage_devices'], cloud_manager=self)
def get_storages(self, storage_type='normal'):
        """
        Return a list of Storage objects from the API.

        Storage types: public, private, normal, backup, cdrom, template, favorite
        """
        res = self.get_request('/storage/' + storage_type)
        return Storage._create_storage_objs(res['storages'], cloud_manager=self)
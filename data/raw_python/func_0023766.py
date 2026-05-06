def prepare_post_body(self):
        """
        Prepare a JSON serializable dict from a Server instance with nested.

        Storage instances.
        """
        body = dict()
        # mandatory
        body['server'] = {
            'hostname': self.hostname,
            'zone': self.zone,
            'title': self.title,
            'storage_devices': {}
        }

        # optional fields

        for optional_field in self.optional_fields:
            if hasattr(self, optional_field):
                body['server'][optional_field] = getattr(self, optional_field)

        # set password_delivery default as 'none' to prevent API from sending
        # emails (with credentials) about each created server
        if not hasattr(self, 'password_delivery'):
            body['server']['password_delivery'] = 'none'

        # collect storage devices and create a unique title (see: Storage.title in API doc)
        # for each of them

        body['server']['storage_devices'] = {
            'storage_device': []
        }

        storage_title_id = 0  # running number for unique storage titles
        for storage in self.storage_devices:
            if not hasattr(storage, 'os') or storage.os is None:
                storage_title_id += 1
            storage_body = storage.to_dict()

            # setup default titles for storages unless the user has specified
            # them at storage.title
            if not hasattr(storage, 'title') or not storage.title:
                if hasattr(storage, 'os') and storage.os:
                    storage_body['title'] = self.hostname + ' OS disk'
                else:
                    storage_body['title'] = self.hostname + ' storage disk ' + str(storage_title_id)


            # figure out the storage `action` parameter
            # public template
            if hasattr(storage, 'os') and storage.os:
                storage_body['action'] = 'clone'
                storage_body['storage'] = OperatingSystems.get_OS_UUID(storage.os)

            # private template
            elif hasattr(storage, 'uuid'):
                storage_body['action'] = 'clone'
                storage_body['storage'] = storage.uuid

            # create a new storage
            else:
                storage_body['action'] = 'create'

            body['server']['storage_devices']['storage_device'].append(storage_body)

        if hasattr(self, 'ip_addresses') and self.ip_addresses:
            body['server']['ip_addresses'] = {
                'ip_address': [
                    ip.to_dict() for ip in self.ip_addresses
                ]
            }


        return body
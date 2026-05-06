def to_dict(self):
        """
        Prepare a JSON serializable dict for read-only purposes.

        Includes storages and IP-addresses.
        Use prepare_post_body for POST and .save() for PUT.
        """
        fields = dict(vars(self).items())

        if self.populated:
            fields['ip_addresses'] = []
            fields['storage_devices'] = []
            for ip in self.ip_addresses:
                fields['ip_addresses'].append({
                    'address': ip.address,
                    'access': ip.access,
                    'family': ip.family
                })

            for storage in self.storage_devices:
                fields['storage_devices'].append({
                    'address': storage.address,
                    'storage': storage.uuid,
                    'storage_size': storage.size,
                    'storage_title': storage.title,
                    'type': storage.type,
                })

        del fields['populated']
        del fields['cloud_manager']
        return fields
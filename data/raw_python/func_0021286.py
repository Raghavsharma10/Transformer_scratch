def custom_resolve(self):
        """ If a custom resolver is defined, perform custom resolution on
        the contained addresses.

        :return:
        """
        if not callable(self.custom_resolver):
            return
        new_addresses = []
        for address in self.addresses:
            for new_address in self.custom_resolver(address):
                new_addresses.append(new_address)
        self.addresses = new_addresses
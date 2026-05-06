def save(self):
        """
        Sync local changes in server's attributes to the API.

        Note: DOES NOT sync IPAddresses and storage_devices,
        use add_ip, add_storage, remove_ip, remove_storage instead.
        """
        # dict comprehension that also works with 2.6
        # http://stackoverflow.com/questions/21069668/alternative-to-dict-comprehension-prior-to-python-2-7
        kwargs = dict(
            (field, getattr(self, field))
            for field in self.updateable_fields
            if hasattr(self, field)
        )

        self.cloud_manager.modify_server(self.uuid, **kwargs)
        self._reset(kwargs)
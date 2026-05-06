def setup_storage(self):
        """Save existing FileField storages and patch them with test instance(s).

        If storage_per_field is False (default) this function will create a
        single instance here and assign it to self.storage to be used for all
        filefields.
        If storage_per_field is True, an independent storage instance will be
        used for each FileField .
        """
        if self.storage_callable is not None and not self.storage_per_field:
            self.storage = self.get_storage_from_callable(field=None)
        super(override_storage, self).setup_storage()
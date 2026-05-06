def generate_and_register_handle(self, prefix, location, checksum=None, additional_URLs=None, **extratypes):
        '''
        Register a new Handle with a unique random name (random UUID).

        :param prefix: The prefix of the handle to be registered. The method
            will generate a suffix.
        :param location: The URL of the data entity to be referenced.
        :param checksum: Optional. The checksum string.
        :param extratypes: Optional. Additional key value pairs as dict.
        :param additional_URLs: Optional. A list of URLs (as strings) to be
            added to the handle record as 10320/LOC entry.
        :raises: :exc:`~b2handle.handleexceptions.HandleAuthenticationError`
        :return: The new handle name.
        '''

        LOGGER.debug('generate_and_register_handle...')

        handle = self.generate_PID_name(prefix)
        handle = self.register_handle(
            handle,
            location,
            checksum,
            additional_URLs,
            overwrite=True,
            **extratypes
        )
        return handle
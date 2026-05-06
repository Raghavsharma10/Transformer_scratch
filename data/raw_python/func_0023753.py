def _reset(self, server, **kwargs):
        """
        Reset the server object with new values given as params.

        - server: a dict representing the server. e.g the API response.
        - kwargs: any meta fields such as cloud_manager and populated.

        Note: storage_devices and ip_addresses may be given in server as dicts or
        in kwargs as lists containing Storage and IPAddress objects.
        """
        if server:
            # handle storage, ip_address dicts and tags if they exist
            Server._handle_server_subobjs(server, kwargs.get('cloud_manager'))

            for key in server:
                object.__setattr__(self, key, server[key])

        for key in kwargs:
            object.__setattr__(self, key, kwargs[key])
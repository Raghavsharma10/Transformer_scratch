def instantiate_with_username_and_password(handle_server_url, username, password, **config):
        '''
        Initialize client against an HSv8 instance with full read/write access.

        The method will throw an exception upon bad syntax or non-existing
        Handle. The existence or validity of the password in the handle is
        not checked at this moment.

        :param handle_server_url: The URL of the Handle System server.
        :param username: This must be a handle value reference in the format
            "index:prefix/suffix".
        :param password: This is the password stored as secret key in the
            actual Handle value the username points to.
        :param \**config: More key-value pairs may be passed that will be passed
            on to the constructor as config.
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`: If the username handle is not found.
        :raises: :exc:`~b2handle.handleexceptions.HandleSyntaxError`
        :return: An instance of the client.
        '''

        inst = EUDATHandleClient(handle_server_url, username=username, password=password, **config)
        return inst
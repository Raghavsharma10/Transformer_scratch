def instantiate_for_read_and_search(handle_server_url, reverselookup_username, reverselookup_password, **config):
        '''
        Initialize client with read access and with search function.

        :param handle_server_url: The URL of the Handle Server. May be None
            (then, the default 'https://hdl.handle.net' is used).
        :param reverselookup_username: The username to authenticate at the
            reverse lookup servlet.
        :param reverselookup_password: The password to authenticate at the
            reverse lookup servlet.
        :param \**config: More key-value pairs may be passed that will be passed
            on to the constructor as config. Config options from the
            credentials object are overwritten by this.
        :return: An instance of the client.
        '''

        if handle_server_url is None and 'reverselookup_baseuri' not in config.keys():
            raise TypeError('You must specify either "handle_server_url" or "reverselookup_baseuri".' + \
                ' Searching not possible without the URL of a search servlet.')

        inst = EUDATHandleClient(
            handle_server_url,
            reverselookup_username=reverselookup_username,
            reverselookup_password=reverselookup_password,
            **config
        )
        return inst
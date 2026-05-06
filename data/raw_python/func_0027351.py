def instantiate_with_credentials(credentials, **config):
        '''
        Initialize the client against an HSv8 instance with full read/write
        access.

        :param credentials: A credentials object, see separate class
            PIDClientCredentials.
        :param \**config: More key-value pairs may be passed that will be passed
            on to the constructor as config. Config options from the
            credentials object are overwritten by this.
        :raises: :exc:`~b2handle.handleexceptions.HandleNotFoundException`: If the username handle is not found.
        :return: An instance of the client.
        '''
        key_value_pairs = credentials.get_all_args()

        if config is not None:
            key_value_pairs.update(**config)  # passed config overrides json file

        inst = EUDATHandleClient(**key_value_pairs)
        return inst
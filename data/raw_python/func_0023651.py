def command_factory(command):
    """A factory which returns functions for direct daemon communication.

    This factory will create a function which sends a payload to the daemon
    and returns the unpickled object which is returned by the daemon.

    Args:
        command (string): The type of payload this should be. This determines
            as what kind of instruction this will be interpreted by the daemon.
    Returns:
        function: The created function.
    """
    def communicate(body={}, root_dir=None):
        """Communicate with the daemon.

        This function sends a payload to the daemon and returns the unpickled
        object sent by the daemon.

        Args:
            body (dir): Any other arguments that should be put into the payload.
            root_dir (str): The root directory in which we expect the daemon.
                            We need this to connect to the daemons socket.
        Returns:
            function: The returned payload.
        """

        client = connect_socket(root_dir)
        body['mode'] = command
        # Delete the func entry we use to call the correct function with argparse
        # as functions can't be pickled and this shouldn't be send to the daemon.
        if 'func' in body:
            del body['func']
        data_string = pickle.dumps(body, -1)
        client.send(data_string)

        # Receive message, unpickle and return it
        response = receive_data(client)
        return response
    return communicate
def create(cls, cli, management_address,
               local_username=None, local_password=None,
               remote_username=None, remote_password=None,
               connection_type=None):
        """
        Configures a remote system for remote replication.

        :param cls: this class.
        :param cli: the rest client.
        :param management_address: the management IP address of the remote
            system.
        :param local_username: administrative username of local system.
        :param local_password: administrative password of local system.
        :param remote_username: administrative username of remote system.
        :param remote_password: administrative password of remote system.
        :param connection_type: `ReplicationCapabilityEnum`. Replication
            connection type to the remote system.
        :return: the newly created remote system.
        """

        req_body = cli.make_body(
            managementAddress=management_address, localUsername=local_username,
            localPassword=local_password, remoteUsername=remote_username,
            remotePassword=remote_password, connectionType=connection_type)

        resp = cli.post(cls().resource_class, **req_body)
        resp.raise_if_err()
        return cls.get(cli, resp.resource_id)
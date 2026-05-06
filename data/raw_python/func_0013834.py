def create_remote_system(self, management_address,
                             local_username=None, local_password=None,
                             remote_username=None, remote_password=None,
                             connection_type=None):
        """
        Configures a remote system for remote replication.

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
        return UnityRemoteSystem.create(self._cli, management_address,
                                        local_username=local_username,
                                        local_password=local_password,
                                        remote_username=remote_username,
                                        remote_password=remote_password,
                                        connection_type=connection_type)
def replicate(self, dst_lun_id, max_time_out_of_sync,
                  replication_name=None, replicate_existing_snaps=None,
                  remote_system=None):
        """
        Creates a replication session with a existing lun as destination.

        :param dst_lun_id: destination lun id.
        :param max_time_out_of_sync: maximum time to wait before syncing the
            source and destination. Value `-1` means the automatic sync is not
            performed. `0` means it is a sync replication.
        :param replication_name: replication name.
        :param replicate_existing_snaps: whether to replicate existing snaps.
        :param remote_system: `UnityRemoteSystem` object. The remote system to
            which the replication is being configured. When not specified, it
            defaults to local system.
        :return: created replication session.
        """

        return UnityReplicationSession.create(
            self._cli, self.get_id(), dst_lun_id, max_time_out_of_sync,
            name=replication_name,
            replicate_existing_snaps=replicate_existing_snaps,
            remote_system=remote_system)
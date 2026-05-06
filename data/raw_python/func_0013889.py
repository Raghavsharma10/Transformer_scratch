def replicate_with_dst_resource_provisioning(self, max_time_out_of_sync,
                                                 dst_pool_id,
                                                 dst_lun_name=None,
                                                 remote_system=None,
                                                 replication_name=None,
                                                 dst_size=None, dst_sp=None,
                                                 is_dst_thin=None,
                                                 dst_tiering_policy=None,
                                                 is_dst_compression=None):
        """
        Creates a replication session with destination lun provisioning.

        :param max_time_out_of_sync: maximum time to wait before syncing the
            source and destination. Value `-1` means the automatic sync is not
            performed. `0` means it is a sync replication.
        :param dst_pool_id: id of pool to allocate destination lun.
        :param dst_lun_name: destination lun name.
        :param remote_system: `UnityRemoteSystem` object. The remote system to
            which the replication is being configured. When not specified, it
            defaults to local system.
        :param replication_name: replication name.
        :param dst_size: destination lun size.
        :param dst_sp: `NodeEnum` value. Default storage processor of
            destination lun.
        :param is_dst_thin: indicates whether destination lun is thin or not.
        :param dst_tiering_policy: `TieringPolicyEnum` value. Tiering policy of
            destination lun.
        :param is_dst_compression: indicates whether destination lun is
            compression enabled or not.
        :return: created replication session.
        """

        dst_size = self.size_total if dst_size is None else dst_size

        dst_resource = UnityResourceConfig.to_embedded(
            name=dst_lun_name, pool_id=dst_pool_id,
            size=dst_size, default_sp=dst_sp,
            tiering_policy=dst_tiering_policy, is_thin_enabled=is_dst_thin,
            is_compression_enabled=is_dst_compression)
        return UnityReplicationSession.create_with_dst_resource_provisioning(
            self._cli, self.get_id(), dst_resource, max_time_out_of_sync,
            remote_system=remote_system, name=replication_name)
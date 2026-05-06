def create(cls, cli, src_resource_id, dst_resource_id,
               max_time_out_of_sync, name=None, members=None,
               auto_initiate=None, hourly_snap_replication_policy=None,
               daily_snap_replication_policy=None,
               replicate_existing_snaps=None, remote_system=None,
               src_spa_interface=None, src_spb_interface=None,
               dst_spa_interface=None, dst_spb_interface=None):
        """
        Creates a replication session.

        :param cli: the rest cli.
        :param src_resource_id: id of the replication source, could be
            lun/fs/cg.
        :param dst_resource_id: id of the replication destination.
        :param max_time_out_of_sync: maximum time to wait before syncing the
            source and destination. Value `-1` means the automatic sync is not
            performed. `0` means it is a sync replication.
        :param name: name of the replication.
        :param members: list of `UnityLunMemberReplication` object. If
            `src_resource` is cg, `lunMemberReplication` list need to pass in
            to this parameter as member lun pairing between source and
            destination cg.
        :param auto_initiate: indicates whether to perform the first
            replication sync automatically.
            True - perform the first replication sync automatically.
            False - perform the first replication sync manually.
        :param hourly_snap_replication_policy: `UnitySnapReplicationPolicy`
            object. The policy for replicating hourly scheduled snaps of the
            source resource.
        :param daily_snap_replication_policy: `UnitySnapReplicationPolicy`
            object. The policy for replicating daily scheduled snaps of the
            source resource.
        :param replicate_existing_snaps: indicates whether or not to replicate
            snapshots already existing on the resource.
        :param remote_system: `UnityRemoteSystem` object. The remote system of
            remote replication.
        :param src_spa_interface: `UnityRemoteInterface` object. The
            replication interface for source SPA.
        :param src_spb_interface: `UnityRemoteInterface` object. The
            replication interface for source SPB.
        :param dst_spa_interface: `UnityRemoteInterface` object. The
            replication interface for destination SPA.
        :param dst_spb_interface: `UnityRemoteInterface` object. The
            replication interface for destination SPB.
        :return: the newly created replication session.
        """

        req_body = cli.make_body(
            srcResourceId=src_resource_id, dstResourceId=dst_resource_id,
            maxTimeOutOfSync=max_time_out_of_sync, members=members,
            autoInitiate=auto_initiate, name=name,
            hourlySnapReplicationPolicy=hourly_snap_replication_policy,
            dailySnapReplicationPolicy=daily_snap_replication_policy,
            replicateExistingSnaps=replicate_existing_snaps,
            remoteSystem=remote_system,
            srcSPAInterface=src_spa_interface,
            srcSPBInterface=src_spb_interface,
            dstSPAInterface=dst_spa_interface,
            dstSPBInterface=dst_spb_interface)

        resp = cli.post(cls().resource_class, **req_body)
        resp.raise_if_err()
        return cls.get(cli, resp.resource_id)
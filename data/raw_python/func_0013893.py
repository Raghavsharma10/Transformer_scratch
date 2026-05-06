def create_with_dst_resource_provisioning(
            cls, cli, src_resource_id, dst_resource_config,
            max_time_out_of_sync, name=None, remote_system=None,
            src_spa_interface=None, src_spb_interface=None,
            dst_spa_interface=None, dst_spb_interface=None,
            dst_resource_element_configs=None, auto_initiate=None,
            hourly_snap_replication_policy=None,
            daily_snap_replication_policy=None, replicate_existing_snaps=None):
        """
        Create a replication session along with destination resource
        provisioning.

        :param cli: the rest cli.
        :param src_resource_id: id of the replication source, could be
            lun/fs/cg.
        :param dst_resource_config: `UnityResourceConfig` object. The user
            chosen config for destination resource provisioning. `pool_id` and
            `size` are required for creation.
        :param max_time_out_of_sync: maximum time to wait before syncing the
            source and destination. Value `-1` means the automatic sync is not
            performed. `0` means it is a sync replication.
        :param name: name of the replication.
        :param remote_system: `UnityRemoteSystem` object. The remote system to
            which the replication is being configured. When not specified, it
            defaults to local system.
        :param src_spa_interface: `UnityRemoteInterface` object. The
            replication interface for source SPA.
        :param src_spb_interface: `UnityRemoteInterface` object. The
            replication interface for source SPB.
        :param dst_spa_interface: `UnityRemoteInterface` object. The
            replication interface for destination SPA.
        :param dst_spb_interface: `UnityRemoteInterface` object. The
            replication interface for destination SPB.
        :param dst_resource_element_configs: List of `UnityResourceConfig`
            objects. The user chose config for each of the member element of
            the destination resource.
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
        :return: the newly created replication session.
        """

        req_body = cli.make_body(
            srcResourceId=src_resource_id,
            dstResourceConfig=dst_resource_config,
            maxTimeOutOfSync=max_time_out_of_sync,
            name=name, remoteSystem=remote_system,
            srcSPAInterface=src_spa_interface,
            srcSPBInterface=src_spb_interface,
            dstSPAInterface=dst_spa_interface,
            dstSPBInterface=dst_spb_interface,
            dstResourceElementConfigs=dst_resource_element_configs,
            autoInitiate=auto_initiate,
            hourlySnapReplicationPolicy=hourly_snap_replication_policy,
            dailySnapReplicationPolicy=daily_snap_replication_policy,
            replicateExistingSnaps=replicate_existing_snaps)

        resp = cli.type_action(
            cls().resource_class,
            'createReplicationSessionWDestResProvisioning',
            **req_body)
        resp.raise_if_err()
        # response is like:
        # "content": {
        #     "id": {
        #         "id": "42949676351_FNM00150600267_xxxx"
        #     }
        session_resp = resp.first_content['id']
        return cls.get(cli, _id=session_resp['id'])
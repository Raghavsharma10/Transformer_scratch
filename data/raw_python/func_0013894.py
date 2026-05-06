def modify(self, max_time_out_of_sync=None, name=None,
               hourly_snap_replication_policy=None,
               daily_snap_replication_policy=None,
               src_spa_interface=None, src_spb_interface=None,
               dst_spa_interface=None, dst_spb_interface=None):
        """
        Modifies properties of a replication session.

        :param max_time_out_of_sync: same as the one in `create` method.
        :param name: same as the one in `create` method.
        :param hourly_snap_replication_policy: same as the one in `create`
            method.
        :param daily_snap_replication_policy: same as the one in `create`
            method.
        :param src_spa_interface: same as the one in `create` method.
        :param src_spb_interface: same as the one in `create` method.
        :param dst_spa_interface: same as the one in `create` method.
        :param dst_spb_interface: same as the one in `create` method.
        """
        req_body = self._cli.make_body(
            maxTimeOutOfSync=max_time_out_of_sync, name=name,
            hourlySnapReplicationPolicy=hourly_snap_replication_policy,
            dailySnapReplicationPolicy=daily_snap_replication_policy,
            srcSPAInterface=src_spa_interface,
            srcSPBInterface=src_spb_interface,
            dstSPAInterface=dst_spa_interface,
            dstSPBInterface=dst_spb_interface)

        resp = self.action('modify', **req_body)
        resp.raise_if_err()
        return resp
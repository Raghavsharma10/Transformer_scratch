def resume(self, force_full_copy=None,
               src_spa_interface=None, src_spb_interface=None,
               dst_spa_interface=None, dst_spb_interface=None):
        """
        Resumes a replication session.

        This can be applied on replication session when it's operational status
        is reported as Failed over, or Paused.

        :param force_full_copy: needed when replication session goes out of
            sync due to a fault.
            True - replicate all data.
            False - replicate changed data only.
        :param src_spa_interface: same as the one in `create` method.
        :param src_spb_interface: same as the one in `create` method.
        :param dst_spa_interface: same as the one in `create` method.
        :param dst_spb_interface: same as the one in `create` method.
        """
        req_body = self._cli.make_body(forceFullCopy=force_full_copy,
                                       srcSPAInterface=src_spa_interface,
                                       srcSPBInterface=src_spb_interface,
                                       dstSPAInterface=dst_spa_interface,
                                       dstSPBInterface=dst_spb_interface)

        resp = self.action('resume', **req_body)
        resp.raise_if_err()
        return resp
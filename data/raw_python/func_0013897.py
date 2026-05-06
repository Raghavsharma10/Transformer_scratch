def failback(self, force_full_copy=None):
        """
        Fails back a replication session.

        This can be applied on a replication session that is failed over. Fail
        back will synchronize the changes done to original destination back to
        original source site and will restore the original direction of
        session.

        :param force_full_copy: indicates whether to sync back all data from
            the destination SP to the source SP during the failback session.
            True - Sync back all data.
            False - Sync back changed data only.
        """
        req_body = self._cli.make_body(forceFullCopy=force_full_copy)
        resp = self.action('failback', **req_body)
        resp.raise_if_err()
        return resp
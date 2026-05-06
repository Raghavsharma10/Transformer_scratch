def failover(self, sync=None, force=None):
        """
        Fails over a replication session.

        :param sync: True - sync the source and destination resources before
            failing over the asynchronous replication session or keep them in
            sync after failing over the synchronous replication session.
            False - don't sync.
        :param force: True - skip pre-checks on file system(s) replication
            sessions of a NAS server when a replication failover is issued from
            the source NAS server.
            False - don't skip pre-checks.
        """
        req_body = self._cli.make_body(sync=sync, force=force)
        resp = self.action('failover', **req_body)
        resp.raise_if_err()
        return resp
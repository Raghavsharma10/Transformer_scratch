def set_session(self, autocommit=None, readonly=None):
        """Sets one or more parameters in the current connection.

        :param autocommit:
            Switch the connection to autocommit mode. With the current
            version, you need to always enable this, because
            :meth:`commit` is not implemented.

        :param readonly:
            Switch the connection to read-only mode.
        """
        props = {}
        if autocommit is not None:
            props['autoCommit'] = bool(autocommit)
        if readonly is not None:
            props['readOnly'] = bool(readonly)
        props = self._client.connection_sync(self._id, props)
        self._autocommit = props.auto_commit
        self._readonly = props.read_only
        self._transactionisolation = props.transaction_isolation
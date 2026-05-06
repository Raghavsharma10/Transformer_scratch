def fail(self) -> None:
        """ for testing purposes only. always fail in commit """

        _debug("fail")

        # on commit carry out action; on rollback reverse rename
        def on_commit(_obj):
            raise_testfailure("commit")

        def on_rollback(_obj):
            raise_testfailure("rollback")

        return self._process(on_commit, on_rollback)
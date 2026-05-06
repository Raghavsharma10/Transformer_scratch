def rollback(self) -> None:
        """
        Roll back to previous database state. However stay inside transaction
        management.
        """
        if len(self._transactions) == 0:
            raise RuntimeError("rollback called outside transaction")

        _debug("rollback:", self._transactions[-1])
        # if something goes wrong here, nothing we can do about it, leave
        # database as is.
        try:
            # for every rollback action ...
            for on_rollback in self._transactions[-1]:
                # execute it
                _debug("--> rolling back", on_rollback)
                self._do_with_retry(on_rollback)
        except:  # noqa: E722
            _debug("--> rollback failed")
            exc_class, exc, tb = sys.exc_info()
            raise tldap.exceptions.RollbackError(
                "FATAL Unrecoverable rollback error: %r" % exc)
        finally:
            # reset everything to clean state
            _debug("--> rollback success")
            self.reset()
def rollback(self):
        """Ignore all changes made in the latest session (terminate the session)."""
        if self.session is not None:
            logger.info("rolling back transaction in %s" % self)
            self.session.close()
            self.session = None
            self.lock_update.release()
        else:
            logger.warning("rollback called but there's no open session in %s" % self)
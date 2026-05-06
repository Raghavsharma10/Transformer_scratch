def open_session(self):
        """
        Open a new session to modify this server.

        You can either call this fnc directly, or turn on autosession which will
        open/commit sessions for you transparently.
        """
        if self.session is not None:
            msg = "session already open; commit it or rollback before opening another one in %s" % self
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("opening a new session")
        logger.info("removing %s" % self.loc_session)
        try:
            shutil.rmtree(self.loc_session)
        except:
            logger.info("failed to delete %s" % self.loc_session)
        logger.info("cloning server from %s to %s" %
                    (self.loc_stable, self.loc_session))
        shutil.copytree(self.loc_stable, self.loc_session)
        self.session = SimServer(self.loc_session, use_locks=self.use_locks)
        self.lock_update.acquire()
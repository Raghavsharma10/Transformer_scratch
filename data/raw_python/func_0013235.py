def count_by_state_unsynced(self, arg):
        """Extends the original object in order to inject checking
        for stalled jobs and killing them if they are running for too long
        """
        if self.kill_timeout is not None:
            self.delete_running(self.kill_timeout)
        return super(KMongoTrials, self).count_by_state_unsynced(arg)
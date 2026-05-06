def unscheduleFirst(self, runnable):
        """
        Remove from given item from the schedule.

        If runnable is scheduled to run multiple times, only the temporally first
        is removed.
        """
        for evt in self.store.query(TimedEvent, TimedEvent.runnable == runnable, sort=TimedEvent.time.ascending):
            evt.deleteFromStore()
            break
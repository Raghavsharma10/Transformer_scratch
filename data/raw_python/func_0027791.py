def scheduledTimes(self, runnable):
        """
        Return an iterable of the times at which the given item is scheduled to
        run.
        """
        events = self.store.query(
            TimedEvent, TimedEvent.runnable == runnable)
        return (event.time for event in events if not event.running)
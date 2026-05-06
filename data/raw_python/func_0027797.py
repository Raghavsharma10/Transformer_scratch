def migrateUp(self):
        """
        Recreate the hooks in the site store to trigger this SubScheduler.
        """
        te = self.store.findFirst(TimedEvent, sort=TimedEvent.time.descending)
        if te is not None:
            self._transientSchedule(te.time, None)
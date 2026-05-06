def _transientSchedule(self, when, now):
        """
        If this service's store is attached to its parent, ask the parent to
        schedule this substore to tick at the given time.

        @param when: The time at which to tick.
        @type when: L{epsilon.extime.Time}

        @param now: Present for signature compatibility with
            L{_SiteScheduler._transientSchedule}, but ignored otherwise.
        """
        if self.store.parent is not None:
            subStore = self.store.parent.getItemByID(self.store.idInParent)
            hook = self.store.parent.findOrCreate(
                _SubSchedulerParentHook,
                subStore=subStore)
            hook._schedule(when)
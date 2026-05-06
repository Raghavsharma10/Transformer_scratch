def _schedule(self, when):
        """
        Ensure that this hook is scheduled to run at or before C{when}.
        """
        sched = IScheduler(self.store)
        for scheduledAt in sched.scheduledTimes(self):
            if when < scheduledAt:
                sched.reschedule(self, scheduledAt, when)
            break
        else:
            sched.schedule(self, when)
def _scheduleMePlease(self):
        """
        This queue needs to have its run() method invoked at some point in the
        future.  Tell the dependent scheduler to schedule it if it isn't
        already pending execution.
        """
        sched = IScheduler(self.store)
        if len(list(sched.scheduledTimes(self))) == 0:
            sched.schedule(self, sched.now())
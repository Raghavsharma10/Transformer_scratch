def run(self):
        """
        Try to run one unit of work through one listener.  If there are more
        listeners or more work, reschedule this item to be run again in
        C{self.busyInterval} milliseconds, otherwise unschedule it.

        @rtype: L{extime.Time} or C{None}
        @return: The next time at which to run this item, used by the scheduler
        for automatically rescheduling, or None if there is no more work to do.
        """
        now = extime.Time()
        if self.step():
            self.scheduled = now + datetime.timedelta(milliseconds=self.busyInterval)
        else:
            self.scheduled = None
        return self.scheduled
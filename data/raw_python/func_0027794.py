def _transientSchedule(self, when, now):
        """
        If the service is currently running, schedule a tick to happen no
        later than C{when}.

        @param when: The time at which to tick.
        @type when: L{epsilon.extime.Time}

        @param now: The current time.
        @type now: L{epsilon.extime.Time}
        """
        if not self.running:
            return
        if self.timer is not None:
            if self.timer.getTime() < when.asPOSIXTimestamp():
                return
            self.timer.cancel()
        delay = when.asPOSIXTimestamp() - now.asPOSIXTimestamp()

        # reactor.callLater allows only positive delay values.  The scheduler
        # may want to have scheduled things in the past and that's OK, since we
        # are dealing with Time() instances it's impossible to predict what
        # they are relative to the current time from user code anyway.
        delay = max(_EPSILON, delay)
        self.timer = self.callLater(delay, self.tick)
        self.nextEventAt = when
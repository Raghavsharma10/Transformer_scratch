def reset(self, secondsFromNow):
        """Reschedule this call for a different time

        @type secondsFromNow: C{float}
        @param secondsFromNow: The number of seconds from the time of the
        C{reset} call at which this call will be scheduled.

        @raise AlreadyCancelled: Raised if this call has been cancelled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            if self.seconds is None:
                new_time = base.seconds() + secondsFromNow
            else:
                new_time = self.seconds() + secondsFromNow
            if new_time < self.time:
                self.delayed_time = 0
                self.time = new_time
                self.resetter(self)
            else:
                self.delayed_time = new_time - self.time
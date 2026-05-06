def stopService(self):
        """
        Stop calling persistent timed events.
        """
        super(_SiteScheduler, self).stopService()
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
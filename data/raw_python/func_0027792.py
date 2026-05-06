def startService(self):
        """
        Start calling persistent timed events whose time has come.
        """
        super(_SiteScheduler, self).startService()
        self._transientSchedule(self.now(), self.now())
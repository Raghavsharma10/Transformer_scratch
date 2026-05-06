def heartbeat(self):
        """
        Watch our counters--as long as things are incrementing, send a ping to
        statuscake sayin we are alive and okay.
        """

        self.thread_debug("heartbeat")

        # check stats -- should be incrementing
        if self.last_stats:
            if self.stats.http_run <= self.last_stats.http_run:
                self.NOTIFY("No monitors run since last heartbeat!", service="heartbeat")
                return
            elif self.stats.http_handled <= self.last_stats.http_handled:
                self.NOTIFY("No monitor results handled since last heartbeat!", service="heartbeat")
                return

        # ping heartbeat as a webhook
        if self.config.get('heartbeat-hook'):
            result = requests.get(self.config.get('heartbeat-hook'))
            if result.status_code != 200:
                self.NOTIFY("Heartbeat ping to statuscake failed!", level="ERROR")

        # keep a static copy of the last run stats
        self.last_stats = self.stats.copy()
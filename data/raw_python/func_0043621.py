def poll_crontab(self):
        """Check crontab and run target jobs
        """
        polled_time = self._get_current_time()
        if polled_time.second >= 30:
            self.log.debug('Skip cronjobs in {}'.format(polled_time))
            return
        for job in self._crontab:
            if not job.is_runnable(polled_time):
                continue
            job.do_action(self, polled_time)
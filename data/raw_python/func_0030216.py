def on_start(self):
        """Runs when the actor is started and schedules a status update
        """
        logger.info('StatusReporter started.')
        # if configured not to report status then return immediately
        if self.config['status_update_interval'] == 0:
            logger.info('StatusReporter disabled by configuration.')
            return
        self.in_future.report_status()
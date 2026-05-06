def report_again(self, current_status):
        """Computes a sleep interval, sleeps for the specified amount of time
        then kicks off another status report.
        """
        # calculate sleep interval based on current status and configured interval
        _m = {'playing': 1, 'paused': 2, 'stopped': 5}[current_status['state']]
        interval = (self.config['status_update_interval'] * _m) / 1000.0
        # sleep for computed interval and kickoff another webhook
        time.sleep(interval)
        self.in_future.report_status()
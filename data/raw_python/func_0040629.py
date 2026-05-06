def _responsive_sleep(self, seconds, wait_log_interval=0, wait_reason=''):
        """When there is litte work to do, the queuing thread sleeps a lot.
        It can't sleep for too long without checking for the quit flag and/or
        logging about why it is sleeping.

        parameters:
            seconds - the number of seconds to sleep
            wait_log_interval - while sleeping, it is helpful if the thread
                                periodically announces itself so that we
                                know that it is still alive.  This number is
                                the time in seconds between log entries.
            wait_reason - the is for the explaination of why the thread is
                          sleeping.  This is likely to be a message like:
                          'there is no work to do'.

        This was also partially motivated by old versions' of Python inability
        to KeyboardInterrupt out of a long sleep()."""

        for x in xrange(int(seconds)):
            self.quit_check()
            if wait_log_interval and not x % wait_log_interval:
                self.logger.info('%s: %dsec of %dsec',
                                 wait_reason,
                                 x,
                                 seconds)
                self.quit_check()
            time.sleep(1.0)
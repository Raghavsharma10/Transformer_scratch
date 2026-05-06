def wait_for_empty_queue(self, wait_log_interval=0, wait_reason=''):
        """Sit around and wait for the queue to become empty

        parameters:
            wait_log_interval - while sleeping, it is helpful if the thread
                                periodically announces itself so that we
                                know that it is still alive.  This number is
                                the time in seconds between log entries.
            wait_reason - the is for the explaination of why the thread is
                          sleeping.  This is likely to be a message like:
                          'there is no work to do'."""
        seconds = 0
        while True:
            if self.task_queue.empty():
                break
            self.quit_check()
            if wait_log_interval and not seconds % wait_log_interval:
                self.logger.info('%s: %dsec so far',
                                 wait_reason,
                                 seconds)
                self.quit_check()
            seconds += 1
            time.sleep(1.0)
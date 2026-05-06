def blocking_start(self, waiting_func=None):
        """this function is just a wrapper around the start and
        wait_for_completion methods.  It starts the queuing thread and then
        waits for it to complete.  If run by the main thread, it will detect
        the KeyboardInterrupt exception (which is what SIGTERM and SIGHUP
        have been translated to) and will order the threads to die."""
        try:
            self.start()
            self.wait_for_completion(waiting_func)
            # it only ends if someone hits  ^C or sends SIGHUP or SIGTERM -
            # any of which will get translated into a KeyboardInterrupt
        except KeyboardInterrupt:
            while True:
                try:
                    self.stop()
                    break
                except KeyboardInterrupt:
                    self.logger.warning('We heard you the first time.  There '
                                   'is no need for further keyboard or signal '
                                   'interrupts.  We are waiting for the '
                                   'worker threads to stop.  If this app '
                                   'does not halt soon, you may have to send '
                                   'SIGKILL (kill -9)')
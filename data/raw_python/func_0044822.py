def __cleanup(self):
        """
        Wait at most twice as long as the given repetition interval
        for the _wrapper_function to terminate.
        
        If after that time the _wrapper_function has not terminated,
        send SIGTERM to and the process.
        
        Wait at most five times as long as the given repetition interval
        for the _wrapper_function to terminate.
        
        If the process still running send SIGKILL automatically if
        auto_kill_on_last_resort was set True or ask the
        user to confirm sending SIGKILL
        """
        # set run to False and wait some time -> see what happens            
        self._run.value = False
        if check_process_termination(proc                     = self._proc,
                                     timeout                  = 2*self.interval,
                                     prefix                   = '',
                                     auto_kill_on_last_resort = self._auto_kill_on_last_resort):
            log.debug("cleanup successful")
        else:
            raise RuntimeError("cleanup FAILED!")
        try:
            self.conn_send.close()
            self._log_queue_listener.stop()
        except OSError:
            pass
        log.debug("wait for monitor thread to join")
        self._monitor_thread.join()
        log.debug("monitor thread to joined")
        self._func_running.value = False
def blocking_start(self, waiting_func=None):
        """this function starts the task manager running to do tasks.  The
        waiting_func is normally used to do something while other threads
        are running, but here we don't have other threads.  So the waiting
        func will never get called.  I can see wanting this function to be
        called at least once after the end of the task loop."""
        self.logger.debug('threadless start')
        try:
            for job_params in self._get_iterator():  # may never raise
                                                     # StopIteration
                self.config.logger.debug('received %r', job_params)
                self.quit_check()
                if job_params is None:
                    if self.config.quit_on_empty_queue:
                        raise KeyboardInterrupt
                    self.logger.info("there is nothing to do.  Sleeping "
                                     "for %d seconds" %
                                     self.config.idle_delay)
                    self._responsive_sleep(self.config.idle_delay)
                    continue
                self.quit_check()
                try:
                    args, kwargs = job_params
                except ValueError:
                    args = job_params
                    kwargs = {}
                try:
                    self.task_func(*args, **kwargs)
                except Exception:
                    self.config.logger.error("Error in processing a job",
                                             exc_info=True)
        except KeyboardInterrupt:
            self.logger.debug('queuingThread gets quit request')
        finally:
            self.quit = True
            self.logger.debug("ThreadlessTaskManager dies quietly")
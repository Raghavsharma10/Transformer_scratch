def _queuing_thread_func(self):
        """This is the function responsible for reading the iterator and
        putting contents into the queue.  It loops as long as there are items
        in the iterator.  Should something go wrong with this thread, or it
        detects the quit flag, it will calmly kill its workers and then
        quit itself."""
        self.logger.debug('_queuing_thread_func start')
        try:
            for job_params in self._get_iterator():  # may never raise
                                                     # StopIteration
                self.config.logger.debug('received %r', job_params)
                if job_params is None:
                    if self.config.quit_on_empty_queue:
                        self.wait_for_empty_queue(
                            wait_log_interval=10,
                            wait_reason='waiting for queue to drain'
                        )
                        raise KeyboardInterrupt
                    self.logger.info("there is nothing to do.  Sleeping "
                                     "for %d seconds" %
                                     self.config.idle_delay)
                    self._responsive_sleep(self.config.idle_delay)
                    continue
                self.quit_check()
                #self.logger.debug("queuing job %s", job_params)
                self.task_queue.put((self.task_func, job_params))
        except Exception:
            self.logger.error('queuing jobs has failed', exc_info=True)
        except KeyboardInterrupt:
            self.logger.debug('queuingThread gets quit request')
        finally:
            self.logger.debug("we're quitting queuingThread")
            self._kill_worker_threads()
            self.logger.debug("all worker threads stopped")
            # now that we've killed all the workers, we can set the quit flag
            # to True.  This will cause any other threads to die and shut down
            # the application.  Originally, the setting of this flag was at the
            # start of this "finally" block.  However, that meant that the
            # workers would abort their currently running jobs.  In the case of
            # of the natural ending of an application where an iterater ran to
            # exhaustion, the workers would die before completing their tasks.
            # Moving the setting of the flag to this location allows the
            # workers to finish and then the app shuts down.
            self.quit = True
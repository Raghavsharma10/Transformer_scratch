def run(self):
        """The main routine for a thread's work.

        The thread pulls tasks from the task queue and executes them until it
        encounters a death token.  The death token is a tuple of two Nones.
        """
        try:
            quit_request_detected = False
            while True:
                function, arguments = self.task_queue.get()
                if function is None:
                    # this allows us to watch the threads die and identify
                    # threads that may be hanging or deadlocked
                    self.config.logger.info('quits')
                    break
                if quit_request_detected:
                    continue
                try:
                    try:
                        args, kwargs = arguments
                    except ValueError:
                        args = arguments
                        kwargs = {}
                    function(*args, **kwargs)  # execute the task
                except Exception:
                    self.config.logger.error("Error in processing a job",
                                             exc_info=True)
                except KeyboardInterrupt:  # TODO: can probably go away
                    self.config.logger.info('quit request detected')
                    quit_request_detected = True
                    #thread.interrupt_main()  # only needed if signal handler
                                             # not registered
        except Exception:
            self.config.logger.critical("Failure in task_queue", exc_info=True)
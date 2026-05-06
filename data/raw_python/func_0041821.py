def _setup_task_manager(self):
        """instantiate the threaded task manager to run the producer/consumer
        queue that is the heart of the processor."""
        self.config.logger.info('installing signal handers')
        # set up the signal handler for dealing with SIGTERM. the target should
        # be this app instance so the signal handler can reach in and set the
        # quit flag to be True.  See the 'respond_to_SIGTERM' method for the
        # more information
        respond_to_SIGTERM_with_logging = partial(
            respond_to_SIGTERM,
            target=self
        )
        signal.signal(signal.SIGTERM, respond_to_SIGTERM_with_logging)
        self.task_manager = \
            self.config.producer_consumer.producer_consumer_class(
                self.config.producer_consumer,
                job_source_iterator=self.source_iterator,
                task_func=self.transform
            )
        self.config.executor_identity = self.task_manager.executor_identity
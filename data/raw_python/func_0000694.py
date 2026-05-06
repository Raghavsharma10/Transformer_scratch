def async_refresh(self, *args, **kwargs):
        """
        Trigger an asynchronous job to refresh the cache
        """
        # We trigger the task with the class path to import as well as the
        # (a) args and kwargs for instantiating the class
        # (b) args and kwargs for calling the 'refresh' method

        try:
            enqueue_task(
                dict(
                    klass_str=self.class_path,
                    obj_args=self.get_init_args(),
                    obj_kwargs=self.get_init_kwargs(),
                    call_args=args,
                    call_kwargs=kwargs
                ),
                task_options=self.task_options
            )
        except Exception:
            # Handle exceptions from talking to RabbitMQ - eg connection
            # refused.  When this happens, we try to run the task
            # synchronously.
            logger.error("Unable to trigger task asynchronously - failing "
                         "over to synchronous refresh", exc_info=True)
            try:
                return self.refresh(*args, **kwargs)
            except Exception as e:
                # Something went wrong while running the task
                logger.error("Unable to refresh data synchronously: %s", e,
                             exc_info=True)
            else:
                logger.debug("Failover synchronous refresh completed successfully")
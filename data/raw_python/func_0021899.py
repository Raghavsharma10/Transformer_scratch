def set_callbacks(self, worker_start_callback: callable, worker_end_callback: callable, are_async: bool = False):
        """
        :param are_async: True if the callbacks execute asynchronously, posting any heavy work to another thread.
        """
        # We are setting self.worker_start_callback and self.worker_end_callback
        # to lambdas instead of saving them in private vars and moving the lambda logic
        # to a member function for, among other reasons, making callback updates atomic,
        # ie. once a callback has been posted, it will be executed as it was in that
        # moment, any call to set_callbacks will only affect callbacks posted since they
        # were updated, but not to any pending callback.

        # If callback is async, execute the start callback in the calling thread
        scheduler = self.immediate if are_async else self.background
        self.worker_start_callback = lambda worker: scheduler(Work(
            lambda: worker_start_callback(worker), "worker_start_callback:" + worker.name
        ))

        # As the end callback is called *just* before the thread dies,
        # there is no problem running it on the thread
        self.worker_end_callback = lambda worker: self.immediate(Work(
            lambda: worker_end_callback(worker), "worker_end_callback:" + worker.name
        ))
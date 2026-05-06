def main(self):
        """this main routine sets up the signal handlers, the source and
        destination crashstorage systems at the  theaded task manager.  That
        starts a flock of threads that are ready to shepherd crashes from
        the source to the destination."""

        self._setup_task_manager()
        self._setup_source_and_destination()
        self.task_manager.blocking_start(waiting_func=self.waiting_func)
        self.close()
        self.config.logger.info('done.')
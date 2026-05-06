def _create_threads(self):
        """
        This method creates job instances.
        """

        creator = JobCreator(
            self.config,
            self.observers.jobs,
            self.logger
        )
        self.jobs = creator.job_factory()
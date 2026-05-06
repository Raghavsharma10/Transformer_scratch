def wait_for_stats(self):
        """Make sure all jobs are finished."""
        logging.debug("waiting for statistics to finish")
        for job in self.stat_jobs:
            job.get()
        sleep(2)
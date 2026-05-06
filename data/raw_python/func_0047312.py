def stop(self):
        """Stop the indicator process."""

        if self.run is True and all([self.job, self.job.is_alive()]):
            print('Done.')
            self.job.terminate()
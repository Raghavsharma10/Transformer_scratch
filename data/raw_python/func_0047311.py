def start(self):
        """Indicate that we are performing work in a thread.

        :returns: multiprocessing job object
        """

        if self.run is True:
            self.job = multiprocessing.Process(target=self.indicator)
            self.job.start()
            return self.job
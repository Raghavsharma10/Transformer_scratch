def nap(self) -> None:
        """
        Go to sleep for the duration of self.delay.

        :returns: None
        """
        self.log.info(f"Sleeping for {self.delay} seconds.")
        for _ in progress.bar(range(self.delay)):
            time.sleep(1)
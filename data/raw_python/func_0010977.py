def start_polling(self, interval):
        """
        Start polling for term updates and streaming.
        """
        interval = float(interval)

        self.polling = True

        # clear the stored list of terms - we aren't tracking any
        self.term_checker.reset()

        logger.info("Starting polling for changes to the track list")
        while self.polling:

            loop_start = time()

            self.update_stream()
            self.handle_exceptions()

            # wait for the interval unless interrupted, compensating for time elapsed in the loop
            elapsed = time() - loop_start
            sleep(max(0.1, interval - elapsed))

        logger.warning("Term poll ceased!")
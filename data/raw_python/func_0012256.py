def stop(self):
        """
            This method stops the brokers interaction with the kafka stream
        """
        self.loop.run_until_complete(self._consumer.stop())
        self.loop.run_until_complete(self._producer.stop())

        # attempt
        try:
            # to cancel the service
            self._consumer_task.cancel()
        # if there was no service
        except AttributeError:
            # keep going
            pass
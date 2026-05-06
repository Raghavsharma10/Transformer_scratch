def start(self):
        """
            This function starts the brokers interaction with the kafka stream
        """
        self.loop.run_until_complete(self._consumer.start())
        self.loop.run_until_complete(self._producer.start())
        self._consumer_task = self.loop.create_task(self._consume_event_callback())
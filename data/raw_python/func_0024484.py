def on_input_queue_declare(self, queue):
        """
        Input queue declaration callback.
        Input Queue/Exchange binding done here

        Args:
            queue: input queue
        """
        self.in_channel.queue_bind(callback=None,
                                   exchange='input_exc',
                                   queue=self.INPUT_QUEUE_NAME,
                                   routing_key="#")
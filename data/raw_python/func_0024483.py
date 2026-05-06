def on_channel_open(self, channel):
        """
        Input channel creation callback
        Queue declaration done here

        Args:
            channel: input channel
        """
        self.in_channel.exchange_declare(exchange='input_exc', type='topic', durable=True)
        channel.queue_declare(callback=self.on_input_queue_declare, queue=self.INPUT_QUEUE_NAME)
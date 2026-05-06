def connect(self):
        """
        make amqp connection and create channels and queue binding
        """
        self.connection = pika.BlockingConnection(BLOCKING_MQ_PARAMS)
        self.client_queue = ClientQueue()
        self.input_channel = self.connection.channel()

        self.input_channel.exchange_declare(exchange=self.INPUT_EXCHANGE,
                                            type='topic',
                                            durable=True)
        self.input_channel.queue_declare(queue=self.INPUT_QUEUE_NAME)
        self.input_channel.queue_bind(exchange=self.INPUT_EXCHANGE, queue=self.INPUT_QUEUE_NAME)
        log.info("Bind to queue named '%s' queue with exchange '%s'" % (self.INPUT_QUEUE_NAME,
                                                                        self.INPUT_EXCHANGE))
def connect(self):
        """
        Creates connection to RabbitMQ server
        """
        if self.connecting:
            log.info('PikaClient: Already connecting to RabbitMQ')
            return

        log.info('PikaClient: Connecting to RabbitMQ')
        self.connecting = True

        self.connection = TornadoConnection(NON_BLOCKING_MQ_PARAMS,
                                            stop_ioloop_on_close=False,
                                            custom_ioloop=self.io_loop,
                                            on_open_callback=self.on_connected)
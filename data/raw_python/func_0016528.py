def declare(self):
        """Declare the exchange.

        Creates the exchange on the broker.

        """
        self.backend.exchange_declare(exchange=self.exchange,
                                      type=self.exchange_type,
                                      durable=self.durable,
                                      auto_delete=self.auto_delete)
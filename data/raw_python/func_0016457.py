def exchange_declare(self, exchange, type, durable, auto_delete):
        """Declare an named exchange."""
        return self.channel.exchange_declare(exchange=exchange,
                                             type=type,
                                             durable=durable,
                                             auto_delete=auto_delete)
def create_exchange(self):
        """
        Creates MQ exchange for this channel
        Needs to be defined only once.
        """
        mq_channel = self._connect_mq()
        mq_channel.exchange_declare(exchange=self.code_name,
                                    exchange_type='fanout',
                                    durable=True)
def create_exchange(self):
        """
        Creates user's private exchange

        Actually user's private channel needed to be defined only once,
        and this should be happened when user first created.
        But since this has a little performance cost,
        to be safe we always call it before binding to the channel we currently subscribe
        """
        channel = self._connect_mq()
        channel.exchange_declare(exchange=self.user.prv_exchange,
                                 exchange_type='fanout',
                                 durable=True)
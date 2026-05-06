def bind_to_channel(self):
        """
        Binds (subscribes) users private exchange to channel exchange
        Automatically called at creation of subscription record.
        """
        if self.channel.code_name != self.user.prv_exchange:
            channel = self._connect_mq()
            channel.exchange_bind(source=self.channel.code_name, destination=self.user.prv_exchange)
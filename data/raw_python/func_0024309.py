def delete_exchange(self):
        """
        Deletes MQ exchange for this channel
        Needs to be defined only once.
        """
        mq_channel = self._connect_mq()
        mq_channel.exchange_delete(exchange=self.code_name)
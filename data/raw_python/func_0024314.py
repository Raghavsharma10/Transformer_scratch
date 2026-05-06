def _republish(self):
        """
        Re-publishes updated message
        """
        mq_channel = self.channel._connect_mq()
        mq_channel.basic_publish(exchange=self.channel.key, routing_key='',
                                 body=json.dumps(self.serialize()))
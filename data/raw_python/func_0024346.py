def send_client_cmd(self, data, cmd=None, via_queue=None):
        """
        Send arbitrary cmd and data to client

        if queue name passed by "via_queue" parameter,
        that queue will be used instead of users private exchange.
        Args:
            data: dict
            cmd: string
            via_queue: queue name,
        """
        mq_channel = self._connect_mq()
        if cmd:
            data['cmd'] = cmd
        if via_queue:
            mq_channel.basic_publish(exchange='',
                                     routing_key=via_queue,
                                     body=json.dumps(data))
        else:
            mq_channel.basic_publish(exchange=self.prv_exchange,
                                     routing_key='',
                                     body=json.dumps(data))
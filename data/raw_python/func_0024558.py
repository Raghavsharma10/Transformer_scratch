def send_to_prv_exchange(self, user_id, message=None):
        """
        Send messages through logged in users private exchange.

        Args:
            user_id string: User key
            message dict: Message object

        """
        exchange = 'prv_%s' % user_id.lower()
        msg = json.dumps(message, cls=ZEngineJSONEncoder)
        log.debug("Sending following users \"%s\" exchange:\n%s " % (exchange, msg))
        self.get_channel().publish(exchange=exchange, routing_key='', body=msg)
def send_to_default_exchange(self, sess_id, message=None):
        """
        Send messages through RabbitMQ's default exchange,
        which will be delivered through routing_key (sess_id).

        This method only used for un-authenticated users, i.e. login process.

        Args:
            sess_id string: Session id
            message dict: Message object.
        """
        msg = json.dumps(message, cls=ZEngineJSONEncoder)
        log.debug("Sending following message to %s queue through default exchange:\n%s" % (
            sess_id, msg))
        self.get_channel().publish(exchange='', routing_key=sess_id, body=msg)
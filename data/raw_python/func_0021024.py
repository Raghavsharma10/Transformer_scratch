def ack(self, msg):
        """Called when a MESSAGE has been received.

        Override this method to handle received messages.

        This function will generate an acknowledge message
        for the given message and transaction (if present).

        """
        message_id = msg['headers']['message-id']
        subscription = msg['headers']['subscription']

        transaction_id = None
        if 'transaction-id' in msg['headers']:
            transaction_id = msg['headers']['transaction-id']

#        print "acknowledging message id <%s>." % message_id

        return ack(message_id, subscription, transaction_id)
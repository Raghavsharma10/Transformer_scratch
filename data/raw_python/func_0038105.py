def on_sanji_message(self, client, userdata, msg):
        """This function will recevie all message from mqtt
        client
            the client instance for this callback
        userdata
            the private user data as set in Client() or userdata_set()
        message
            an instance of MQTTMessage. This is a class with members topic,
            payload, qos, retain.
        """
        try:
            message = Message(msg.payload)
        except (TypeError, ValueError) as e:
            _logger.error(e, exc_info=True)
            return

        if message.type() == MessageType.UNKNOWN:
            _logger.debug("Got an UNKNOWN message, don't dispatch")
            return

        if message.type() == MessageType.RESPONSE:
            self.res_queue.put(message)

        if message.type() == MessageType.REQUEST or \
           message.type() == MessageType.DIRECT or \
           message.type() == MessageType.HOOK or \
           message.type() == MessageType.EVENT:
            self.req_queue.put(message)
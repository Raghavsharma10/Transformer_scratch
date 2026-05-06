def cmd_fire(self, connection, sender, target, payload):
        """
        Sends a message
        """
        msg_target, topic, content = self.parse_payload(payload)

        def callback(sender, payload):
            logging.info("FIRE ACK from %s", sender)

        self.__herald.fire(msg_target, topic, content, callback)
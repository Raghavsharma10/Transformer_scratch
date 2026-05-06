def cmd_notice(self, connection, sender, target, payload):
        """
        Sends a message
        """
        msg_target, topic, content = self.parse_payload(payload)

        def callback(sender, payload):
            logging.info("NOTICE ACK from %s: %s", sender, payload)

        self.__herald.notice(msg_target, topic, content, callback)
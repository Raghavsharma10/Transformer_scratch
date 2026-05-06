def cmd_post(self, connection, sender, target, payload):
        """
        Sends a message
        """
        msg_target, topic, content = self.parse_payload(payload)

        def callback(sender, payload):
            logging.info("POST RES from %s: %s", sender, payload)

        self.__herald.post(msg_target, topic, content, callback)
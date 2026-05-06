def cmd_send(self, connection, sender, target, payload):
        """
        Sends a message
        """
        msg_target, topic, content = self.parse_payload(payload)
        results = self.__herald.send(msg_target, topic, content)
        self.safe_send(connection, target, "GOT RESULT: {0}".format(results))
def _ack(self, sender, uid, level, payload=None):
        """
        Replies to a message
        """
        content = {'reply-to': uid,
                   'reply-level': level,
                   'payload': payload}
        self.__client.send_message(sender, json.dumps(content))
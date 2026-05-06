def fire(self, target, topic, content, callback=None):
        """
        Fires a message
        """
        message = self.__make_message(topic, content)
        if callback is not None:
            self.__callbacks[message['uid']] = ('fire', callback)

        self.__client.send_message(target, json.dumps(message), message['uid'])
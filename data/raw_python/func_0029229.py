def chat_update_message(self, channel, text, timestamp, **params):
        """chat.update

        This method updates a message.

        Required parameters:
            `channel`: Channel containing the message to be updated. (e.g: "C1234567890")
            `text`: New text for the message, using the default formatting rules. (e.g: "Hello world")
            `timestamp`:  Timestamp of the message to be updated (e.g: "1405894322.002768")

        https://api.slack.com/methods/chat.update
        """
        method = 'chat.update'
        if self._channel_is_name(channel):
            # chat.update only takes channel ids (not channel names)
            channel = self.channel_name_to_id(channel)
        params.update({
            'channel': channel,
            'text': text,
            'ts': timestamp,
        })
        return self._make_request(method, params)
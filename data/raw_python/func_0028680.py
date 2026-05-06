def send_msg(self, text, channel, confirm=True):
        """
        Send a message to a channel or group via Slack RTM socket, returning
        the resulting message object

        params:
         - text(str): Message text to send
         - channel(Channel): Target channel
         - confirm(bool): If True, wait for a reply-to confirmation before returning.
        """

        self._send_id += 1
        msg = SlackMsg(self._send_id, channel.id, text)
        self.ws.send(msg.json)
        self._stats['messages_sent'] += 1

        if confirm:
            # Wait for confirmation our message was received
            for e in self.events():
                if e.get('reply_to') == self._send_id:
                    msg.sent = True
                    msg.ts = e.ts
                    return msg
        else:
            return msg
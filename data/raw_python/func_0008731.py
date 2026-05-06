def send_message(self, body, subject=None, reply_to=None):
        """
        Send a message to this user from the logged in user.

        :param body: The body of the message.
        :param subject: The subject of the message. Note that if the this
            message is a reply, then the subject of the first message will be
            used instead.
        :param reply_to: Messages can either be replies to other messages or
            start a new message thread. If this is None it will start a new
            message thread. If it's a Message object or message_id, then the
            new message will be sent as a reply to the reply_to message.
        """
        url = self._imgur._base_url + "/3/message"
        parent_id = reply_to.id if isinstance(reply_to, Message) else reply_to
        payload = {'recipient': self.name, 'body': body, 'subject': subject,
                   'parent_id': parent_id}
        self._imgur._send_request(url, params=payload, needs_auth=True,
                                  method='POST')
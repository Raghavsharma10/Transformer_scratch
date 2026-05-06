def reply(self, body):
        """
        Reply to this message.

        This is a convenience method calling User.send_message. See it for more
        information on usage. Note that both recipient and reply_to are given
        by using this convenience method.

        :param body: The body of the message.
        """
        return self.author.send_message(body=body, reply_to=self.id)
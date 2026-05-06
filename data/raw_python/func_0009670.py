def post(self, text=None, attachments=None, source_guid=None):
        """Post a direct message to the user.

        :param str text: the message content
        :param attachments: message attachments
        :param str source_guid: a client-side unique ID for the message
        :return: the message sent
        :rtype: :class:`~groupy.api.messages.DirectMessage`
        """
        return self.messages.create(text=text, attachments=attachments,
                                    source_guid=source_guid)
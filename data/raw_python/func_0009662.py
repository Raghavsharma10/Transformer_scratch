def create(self, text=None, attachments=None, source_guid=None):
        """Create a new message in the group.

        :param str text: the text of the message
        :param attachments: a list of attachments
        :type attachments: :class:`list`
        :param str source_guid: a unique identifier for the message
        :return: the created message
        :rtype: :class:`~groupy.api.messages.Message`
        """
        message = {
            'source_guid': source_guid or str(time.time()),
        }

        if text is not None:
            message['text'] = text

        if attachments is not None:
            message['attachments'] = [a.to_json() for a in attachments]

        payload = {'message': message}
        response = self.session.post(self.url, json=payload)
        message = response.data['message']
        return Message(self, **message)
def post(self, bot_id, text, attachments=None):
        """Post a new message as a bot to its room.

        :param str bot_id: the ID of the bot
        :param str text: the text of the message
        :param attachments: a list of attachments
        :type attachments: :class:`list`
        :return: ``True`` if successful
        :rtype: bool
        """
        url = utils.urljoin(self.url, 'post')
        payload = dict(bot_id=bot_id, text=text)

        if attachments:
            payload['attachments'] = [a.to_json() for a in attachments]

        response = self.session.post(url, json=payload)
        return response.ok
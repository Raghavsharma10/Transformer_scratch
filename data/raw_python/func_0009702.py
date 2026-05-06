def post(self, text, attachments=None):
        """Post a message as the bot.

        :param str text: the text of the message
        :param attachments: a list of attachments
        :type attachments: :class:`list`
        :return: ``True`` if successful
        :rtype: bool
        """
        return self.manager.post(self.bot_id, text, attachments)
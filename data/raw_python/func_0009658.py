def list_since(self, message_id, limit=None):
        """Return a page of group messages created since a message.

        This is used to fetch the most recent messages after another. There
        may exist messages between the one given and the ones returned. Use
        :func:`list_after` to retrieve newer messages without skipping any.

        :param str message_id: the ID of a message
        :param int limit: maximum number of messages per page
        :return: group messages
        :rtype: :class:`~groupy.pagers.MessageList`
        """
        return self.list(since_id=message_id, limit=limit)
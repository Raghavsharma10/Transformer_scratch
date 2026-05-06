def list_all_before(self, message_id, limit=None):
        """Return all group messages created before a message.

        :param str message_id: the ID of a message
        :param int limit: maximum number of messages per page
        :return: group messages
        :rtype: generator
        """
        return self.list_before(message_id, limit=limit).autopage()
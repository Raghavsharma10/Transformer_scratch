def list_all_after(self, message_id, limit=None):
        """Return all group messages created after a message.

        :param str message_id: the ID of a message
        :param int limit: maximum number of messages per page
        :return: group messages
        :rtype: generator
        """
        return self.list_after(message_id, limit=limit).autopage()
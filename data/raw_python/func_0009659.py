def list_after(self, message_id, limit=None):
        """Return a page of group messages created after a message.

        This is used to page forwards through messages.

        :param str message_id: the ID of a message
        :param int limit: maximum number of messages per page
        :return: group messages
        :rtype: :class:`~groupy.pagers.MessageList`
        """
        return self.list(after_id=message_id, limit=limit)
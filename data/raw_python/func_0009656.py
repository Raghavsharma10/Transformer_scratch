def list(self, page=1, per_page=10):
        """List a page of chats.

        :param int page: which page
        :param int per_page: how many chats per page
        :return: chats with other users
        :rtype: :class:`~groupy.pagers.ChatList`
        """
        return pagers.ChatList(self, self._raw_list, per_page=per_page,
                               page=page)
def create_bot(self, name, avatar_url=None, callback_url=None, dm_notification=None,
                   **kwargs):
        """Create a new bot in a particular group.

        :param str name: bot name
        :param str avatar_url: the URL of an image to use as an avatar
        :param str callback_url: a POST-back URL for each new message
        :param bool dm_notification: whether to POST-back for direct messages?
        :return: the new bot
        :rtype: :class:`~groupy.api.bots.Bot`
        """
        return self._bots.create(name=name, group_id=self.group_id,
                                 avatar_url=avatar_url, callback_url=callback_url,
                                 dm_notification=dm_notification)
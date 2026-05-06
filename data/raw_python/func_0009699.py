def create(self, name, group_id, avatar_url=None, callback_url=None,
               dm_notification=None, **kwargs):
        """Create a new bot in a particular group.

        :param str name: bot name
        :param str group_id: the group_id of a group
        :param str avatar_url: the URL of an image to use as an avatar
        :param str callback_url: a POST-back URL for each new message
        :param bool dm_notification: whether to POST-back for direct messages?
        :return: the new bot
        :rtype: :class:`~groupy.api.bots.Bot`
        """
        payload = {
            'bot': {
                'name': name,
                'group_id': group_id,
                'avatar_url': avatar_url,
                'callback_url': callback_url,
                'dm_notification': dm_notification,
            },
        }
        payload['bot'].update(kwargs)
        response = self.session.post(self.url, json=payload)
        bot = response.data['bot']
        return Bot(self, **bot)
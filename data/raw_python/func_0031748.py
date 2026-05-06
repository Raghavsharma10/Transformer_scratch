def get_chat_server(self, channel):
        """Get an appropriate chat server for the given channel

        Usually the server is irc.twitch.tv. But because of the delicate
        twitch chat, they use a lot of servers. Big events are on special
        event servers. This method tries to find a good one.

        :param channel: the channel with the chat
        :type channel: :class:`models.Channel`
        :returns: the server address and port
        :rtype: (:class:`str`, :class:`int`)
        :raises: None
        """
        r = self.oldapi_request(
            'GET', 'channels/%s/chat_properties' % channel.name)
        json = r.json()
        servers = json['chat_servers']

        try:
            r = self.get(TWITCH_STATUSURL)
        except requests.HTTPError:
            log.debug('Error getting chat server status. Using random one.')
            address = servers[0]
        else:
            stats = [client.ChatServerStatus(**d) for d in r.json()]
            address = self._find_best_chat_server(servers, stats)

        server, port = address.split(':')
        return server, int(port)
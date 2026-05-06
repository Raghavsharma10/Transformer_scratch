def send_raw(self, string):
        """Send raw string to the server.

        The string will be padded with appropriate CR LF.
        If too many messages are sent, this will call
        :func:`time.sleep` until it is allowed to send messages again.

        :param string: the raw string to send
        :type string: :class:`str`
        :returns: None
        :raises: :class:`irc.client.InvalidCharacters`,
                 :class:`irc.client.MessageTooLong`,
                 :class:`irc.client.ServerNotConnectedError`
        """
        waittime = self.get_waittime()
        if waittime:
            log.debug('Sent too many messages. Waiting %s seconds',
                      waittime)
            time.sleep(waittime)
        return super(ServerConnection3, self).send_raw(string)
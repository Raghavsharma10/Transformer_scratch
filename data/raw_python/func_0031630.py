def block_user(self, username, domain):
        """Block the specified user.

        The default implementation calls :py:func:`~xmpp_backends.base.XmppBackendBase.set_password` with a
        random password.

        :param username: The username of the user.
        :type  username: str
        :param   domain: The domain of the user.
        :type    domain: str
        """
        self.set_password(username, domain, self.get_random_password())
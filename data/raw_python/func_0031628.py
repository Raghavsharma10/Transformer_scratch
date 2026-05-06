def create_reservation(self, username, domain, email=None):
        """Reserve a new account.

        This method is called when a user account should be reserved, meaning that the account can no longer
        be registered by anybody else but the user cannot yet log in either. This is useful if e.g. an email
        confirmation is still pending.

        The default implementation calls :py:func:`~xmpp_backends.base.XmppBackendBase.create_user` with a
        random password.

        :param username: The username of the user.
        :type  username: str
        :param   domain: The domain of the user.
        :type    domain: str
        :param    email: The email address provided by the user. Note that at this point it is not confirmed.
            You are free to ignore this parameter.
        """
        password = self.get_random_password()
        self.create(username=username, domain=domain, password=password, email=email)
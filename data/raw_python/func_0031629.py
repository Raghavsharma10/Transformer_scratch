def confirm_reservation(self, username, domain, password, email=None):
        """Confirm a reservation for a username.

        The default implementation just calls :py:func:`~xmpp_backends.base.XmppBackendBase.set_password` and
        optionally :py:func:`~xmpp_backends.base.XmppBackendBase.set_email`.
        """
        self.set_password(username=username, domain=domain, password=password)
        if email is not None:
            self.set_email(username=username, domain=domain, email=email)
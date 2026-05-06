def set_password(self, raw_password):
        """Calls :py:func:`~xmpp_backends.base.XmppBackendBase.set_password` for the user.

        If password is ``None``, calls :py:func:`~xmpp_backends.base.XmppBackendBase.set_unusable_password`.
        """
        if raw_password is None:
            self.set_unusable_password()
        else:
            xmpp_backend.set_password(self.node, self.domain, raw_password)
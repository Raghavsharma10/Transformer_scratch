def check_password(self, raw_password):
        """Calls :py:func:`~xmpp_backends.base.XmppBackendBase.check_password` for the user."""
        return xmpp_backend.check_password(self.node, self.domain, raw_password)
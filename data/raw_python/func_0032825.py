def _verifySender(self, sender):
        """
        Verify that this sender is valid.
        """
        if self.store.findFirst(
            LoginMethod,
            AND(LoginMethod.localpart == sender.localpart,
                LoginMethod.domain == sender.domain,
                LoginMethod.internal == True)) is None:
            raise BadSender(sender.localpart + u'@' + sender.domain,
                            [lm.localpart + u'@' + lm.domain
                             for lm in self.store.query(
                        LoginMethod, LoginMethod.internal == True)])
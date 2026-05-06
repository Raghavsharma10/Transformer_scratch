def get_user(self, id=None, name=None, email=None):
        """ Get user object by email or id.
        """
        log.info("Picking user: %s (%s) (%s)" % (name, email, id))
        from qubell.api.private.user import User
        if email:
            user = User.get(self._router, organization=self, email=email)
        else:
            user = self.users[id or name]
        return user
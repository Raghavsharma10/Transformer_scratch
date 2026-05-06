def add_user(self, user):
        """Adds a user to the channel."""
        if not isinstance(user, User):
            user = User(user)
        if user.nick in self.members:
            _log.warning("Ignoring request to add user '%s' to channel '%s' "
                         "because that user is already in the member list.",
                         user, self.name)
            return
        self.members[user.nick] = user
        _log.debug("Added '%s' to channel '%s'", user, self.name)
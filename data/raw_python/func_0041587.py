def find_users_by_email(self, email, user_base='active'):
        """Return list of users with given email address"""
        users = []
        for user in getattr(self, 'users')(user_base).values():
            mail = user.mail
            if mail and email in mail:
                users.append(user)
        log.debug('%s users with email address %s: %s' % (user_base.capitalize(), email, len(users)))
        return users
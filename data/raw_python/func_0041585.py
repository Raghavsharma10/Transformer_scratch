def users(self, user_base='active'):
        """Return dict of users"""
        if not getattr(self, '_%s_users' % user_base):
            self._get_users(user_base)
        return getattr(self, '_%s_users' % user_base)
def describe_users(self, users_filter=None):
        """Return a list of users matching a filter (if provided)."""
        user_list = Users(oktypes=User)
        for user in self._user_list:
            if users_filter and (users_filter.get('name') == user.name or users_filter.get('uid') == user.uid):
                user_list.append(user)
        return user_list
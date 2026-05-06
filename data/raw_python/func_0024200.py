def set_user(self, user):
        """
        Writes user data to session.

        Args:
            user: User object

        """
        self.session['user_id'] = user.key
        self.session['user_data'] = user.clean_value()
        role = self.get_role()
        # TODO: this should be remembered from previous login
        # self.session['role_data'] = default_role.clean_value()
        self.session['role_id'] = role.key
        self.current.role_id = role.key
        self.current.user_id = user.key
        # self.perm_cache = PermissionCache(role.key)
        self.session['permissions'] = role.get_permissions()
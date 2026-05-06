def get_permissions(self):
        """
        Permissions of the user.

        Returns:
            List of Permission objects.
        """
        user_role = self.last_login_role() if self.last_login_role_key else self.role_set[0].role
        return user_role.get_permissions()
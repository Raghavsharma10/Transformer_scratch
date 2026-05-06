def change_user_role(self):
        """
        Changes user's role from current role to chosen role.
        """

        # Get chosen role_key from user form.
        role_key = self.input['form']['role_options']
        # Assign chosen switch role key to user's last_login_role_key field
        self.current.user.last_login_role_key = role_key
        self.current.user.save()
        auth = AuthBackend(self.current)
        # According to user's new role, user's session set again.
        auth.set_user(self.current.user)
        # Dashboard is reloaded according to user's new role.
        self.current.output['cmd'] = 'reload'
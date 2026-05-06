def which_users_can(self, name):
        """Which role can SendMail? """
        _roles = self.which_roles_can(name)
        result =  [self.get_role_members(i.get('role')) for i in _roles]
        return result
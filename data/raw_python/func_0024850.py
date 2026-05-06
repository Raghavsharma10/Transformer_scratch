def add_user(self, user_name, role='user'):
        """
        Calls CF's associate user with org. Valid roles include `user`, `auditor`,
        `manager`,`billing_manager`
        """
        role_uri = self._get_role_uri(role=role)
        return self.api.put(path=role_uri, data={'username': user_name})
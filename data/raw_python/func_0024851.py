def remove_user(self, user_name, role):
        """
        Calls CF's remove user with org
        """
        role_uri = self._get_role_uri(role=role)
        return self.api.delete(path=role_uri, data={'username': user_name})
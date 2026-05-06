def create_role(self, name=None, permissions=""):
        """ Creates role """
        name = name or "autocreated-role"
        from qubell.api.private.role import Role
        return Role.new(self._router, organization=self, name=name, permissions=permissions)
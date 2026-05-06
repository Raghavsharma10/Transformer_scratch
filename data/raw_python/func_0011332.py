def get_role(self, role):
        """Returns a role object
        """
        role = AuthGroup.objects(role=role, creator=self.client).first()
        return role
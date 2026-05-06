def add_role(self, role, description=None):
        """ Creates a new group """
        new_group = AuthGroup(role=role, creator=self.client)
        try:
            new_group.save()
            return True
        except NotUniqueError:
            return False
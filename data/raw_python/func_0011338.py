def add_permission(self, role, name):
        """ authorize a group for something """
        if self.has_permission(role, name):
            return True
        targetGroup = AuthGroup.objects(role=role, creator=self.client).first()
        if not targetGroup:
            return False
        # Create or update
        permission = AuthPermission.objects(name=name).update(
                add_to_set__groups=[targetGroup], creator=self.client, upsert=True
        )
        return True
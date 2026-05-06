def del_permission(self, role, name):
        """ revoke authorization of a group """
        if not self.has_permission(role, name):
            return True
        targetGroup = AuthGroup.objects(role=role, creator=self.client).first()
        target = AuthPermission.objects(groups=targetGroup, name=name, creator=self.client).first()
        if not target:
            return True
        target.delete()
        return True
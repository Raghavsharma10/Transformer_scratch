def user_has_permission(self, user, name):
        """ verify user has permission """
        targetRecord = AuthMembership.objects(creator=self.client, user=user).first()
        if not targetRecord:
            return False
        for group in targetRecord.groups:
            if self.has_permission(group.role, name):
                return True
        return False
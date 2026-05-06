def has_membership(self, user, role):
        """ checks if user is member of a group"""
        targetRecord = AuthMembership.objects(creator=self.client, user=user).first()
        if targetRecord:
            return role in [i.role for i in targetRecord.groups]
        return False
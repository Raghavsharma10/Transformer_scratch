def del_membership(self, user, role):
        """  dismember user from a group """
        if not self.has_membership(user, role):
            return True
        targetRecord = AuthMembership.objects(creator=self.client, user=user).first()
        if not targetRecord:
            return True
        for group in targetRecord.groups:
            if group.role==role:
                targetRecord.groups.remove(group)
        targetRecord.save()
        return True
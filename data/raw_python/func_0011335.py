def add_membership(self, user, role):
        """ make user a member of a group """
        targetGroup = AuthGroup.objects(role=role, creator=self.client).first()
        if not targetGroup:
            return False

        target = AuthMembership.objects(user=user, creator=self.client).first()
        if not target:
            target = AuthMembership(user=user, creator=self.client)

        if not role in [i.role for i in target.groups]:
            target.groups.append(targetGroup)
            target.save()
        return True
def get_user_permissions(self, user):
        """get permissions of a user"""
        memberShipRecords = AuthMembership.objects(creator=self.client, user=user).only('groups')
        results = []
        for each in memberShipRecords:
            for group in each.groups:
                targetPermissionRecords = AuthPermission.objects(creator=self.client,
                                            groups=group).only('name')

                for each_permission in targetPermissionRecords:
                    results.append({'name':each_permission.name})
        return results
def get_user_roles(self, user):
        """get permissions of a user"""
        memberShipRecords = AuthMembership.objects(creator=self.client, user=user).only('groups')
        results = []
        for each in memberShipRecords:
            for group in each.groups:
                results.append({'role':group.role})
        return results
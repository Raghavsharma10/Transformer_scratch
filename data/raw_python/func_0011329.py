def get_role_members(self, role):
        """get permissions of a user"""
        targetRoleDb = AuthGroup.objects(creator=self.client, role=role)
        members = AuthMembership.objects(groups__in=targetRoleDb).only('user')
        return json.loads(members.to_json())
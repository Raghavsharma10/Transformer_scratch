def get_permissions(self, role):
        """gets permissions of role"""
        target_role = AuthGroup.objects(role=role, creator=self.client).first()
        if not target_role:
            return '[]'
        targets = AuthPermission.objects(groups=target_role, creator=self.client).only('name')
        return json.loads(targets.to_json())
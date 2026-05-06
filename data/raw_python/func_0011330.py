def which_roles_can(self, name):
        """Which role can SendMail? """
        targetPermissionRecords = AuthPermission.objects(creator=self.client, name=name).first()
        return [{'role': group.role} for group in targetPermissionRecords.groups]
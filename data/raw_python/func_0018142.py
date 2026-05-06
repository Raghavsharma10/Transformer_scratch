def get_permissions_for_registration(self):
        """
        Utilised by Wagtail's 'register_permissions' hook to allow permissions
        for a all models grouped by this class to be assigned to Groups in
        settings.
        """
        qs = Permission.objects.none()
        for instance in self.modeladmin_instances:
            qs = qs | instance.get_permissions_for_registration()
        return qs
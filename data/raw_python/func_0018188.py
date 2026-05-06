def has_any_permissions(self, user):
        """
        Return a boolean to indicate whether the supplied user has any
        permissions at all on the associated model
        """
        for perm in self.get_all_model_permissions():
            if self.has_specific_permission(user, perm.codename):
                return True
        return False
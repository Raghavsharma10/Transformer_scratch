def add_permission(self, perm):
        """
        Soyut Role Permission nesnesi tanımlamayı sağlar.

        Args:
            perm (object):

        """
        self.Permissions(permission=perm)
        PermissionCache.flush()
        self.save()
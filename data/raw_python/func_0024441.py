def has_permission(self, perm):
        """
        Checks if current user (or role) has the given permission.

        Args:
            perm: Permmission code or object.
             Depends on the :attr:`~zengine.auth.auth_backend.AuthBackend` implementation.

        Returns:
            Boolean.
        """
        return self.user.superuser or self.auth.has_permission(perm)
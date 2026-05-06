def has_permission(self, request, extra_permission=None):
        """
        Returns True if the given HttpRequest has permission to view
        *at least one* page in the admin site.
        """
        permission = request.user.is_active and request.user.is_staff
        if extra_permission:
            permission = permission and request.user.has_perm(extra_permission)
        return permission
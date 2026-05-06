def has_permission(self, request, *args, **kwargs):
        """
        Figures out if the current user has permissions for this view.
        """
        self.kwargs = kwargs
        self.args = args
        self.request = request

        if not getattr(self, 'permission', None):
            return True
        else:
            return request.user.has_perm(self.permission)
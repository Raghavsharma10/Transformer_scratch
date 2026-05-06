def has_publish_permission(self, request, obj=None):
        """
        Returns a boolean if the user in the request has publish permission for the object.
        """
        permission_name = '{}.publish_{}'.format(self.opts.app_label, self.opts.model_name)
        has_permission = request.user.has_perm(permission_name)

        if obj is not None and has_permission is False:
            has_permission = request.user.has_perm(permission_name, obj=obj)

        return has_permission
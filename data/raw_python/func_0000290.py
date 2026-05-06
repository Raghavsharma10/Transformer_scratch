def has_edit_permission(self, request, obj=None, version=None):
        """
        Returns a boolean if the user in the request has edit permission for the object.

        Can also be passed a version object to check if the user has permission to edit a version
        of the object (if they own it).
        """
        # Has the edit permission for this object type
        permission_name = '{}.edit_{}'.format(self.opts.app_label, self.opts.model_name)
        has_permission = request.user.has_perm(permission_name)

        if obj is not None and has_permission is False:
            has_permission = request.user.has_perm(permission_name, obj=obj)

        if has_permission and version is not None:
            # Version must not be saved, and must belong to this user
            if version.version_number or version.owner != request.user:
                has_permission = False

        return has_permission
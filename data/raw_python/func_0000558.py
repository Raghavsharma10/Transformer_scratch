def has_glitter_edit_permission(self, request, obj):
        """
        Return a boolean if a user has edit access to the glitter object/page this object is on.
        """

        # We're testing for the edit permission here with the glitter object - not the current
        # object, not the change permission. Once a user has edit access to an object they can edit
        # all content on it.
        permission_name = '{}.edit_{}'.format(
            obj._meta.app_label, obj._meta.model_name,
        )
        has_permission = (
            request.user.has_perm(permission_name) or
            request.user.has_perm(permission_name, obj=obj)
        )
        return has_permission
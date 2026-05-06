def has_change_permission(self):
        """
        Returns a boolean if the current user has permission to change the current object being
        viewed/edited.
        """
        has_permission = False

        if self.user is not None:
            # We check for the object level permission here, even though by default the Django
            # admin doesn't. If the Django ModelAdmin is extended to allow object level
            # permissions - then this will work as expected.
            permission_name = '{}.change_{}'.format(self.opts.app_label, self.opts.model_name)
            has_permission = (
                self.user.has_perm(permission_name) or
                self.user.has_perm(permission_name, obj=self.obj)
            )

        return has_permission
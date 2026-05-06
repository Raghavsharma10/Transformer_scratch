def has_add_permission(self):
        """
        Returns a boolean if the current user has permission to add another object of the same
        type which is being viewed/edited.
        """
        has_permission = False

        if self.user is not None:
            # We don't check for the object level permission - as the add permission doesn't make
            # sense on a per object level here.
            has_permission = self.user.has_perm(
                '{}.add_{}'.format(self.opts.app_label, self.opts.model_name)
            )

        return has_permission
def can_view(self, user):
        """
        Returns True if user has permission to render this view.

        At minimum this requires an active staff user. If the required_groups
        attribute is not empty then the user must be a member of at least one
        of those groups. If there are no required groups set for the view but
        required groups are set for the bundle then the user must be a member
        of at least one of those groups. If there are no groups to check this
        will return True.
        """

        if user.is_staff and user.is_active:
            if user.is_superuser:
                return True
            elif self.required_groups:
                return self._user_in_groups(user, self.required_groups)
            elif self.bundle.required_groups:
                return self._user_in_groups(user, self.bundle.required_groups)
            else:
                return True

        return False
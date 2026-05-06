def can_user_update_settings(request, view, obj=None):
        """ Only staff can update shared settings, otherwise user has to be an owner of the settings."""
        if obj is None:
            return

        # TODO [TM:3/21/17] clean it up after WAL-634. Clean up service settings update tests as well.
        if obj.customer and not obj.shared:
            return permissions.is_owner(request, view, obj)
        else:
            return permissions.is_staff(request, view, obj)
def _require_staff_for_shared_settings(request, view, obj=None):
        """ Allow to execute action only if service settings are not shared or user is staff """
        if obj is None:
            return

        if obj.settings.shared and not request.user.is_staff:
            raise PermissionDenied(_('Only staff users are allowed to import resources from shared services.'))
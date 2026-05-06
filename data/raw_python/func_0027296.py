def get_permission_checks(self, request, view):
        """
        Get permission checks that will be executed for current action.
        """
        if view.action is None:
            return []
        # if permissions are defined for view directly - use them.
        if hasattr(view, view.action + '_permissions'):
            return getattr(view, view.action + '_permissions')
        # otherwise return view-level permissions + extra view permissions
        extra_permissions = getattr(view, view.action + 'extra_permissions', [])
        if request.method in SAFE_METHODS:
            return getattr(view, 'safe_methods_permissions', []) + extra_permissions
        else:
            return getattr(view, 'unsafe_methods_permissions', []) + extra_permissions
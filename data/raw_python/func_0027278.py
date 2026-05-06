def create_view(self, callback, method, request=None):
        """
        Given a callback, return an actual view instance.
        """
        view = super(WaldurSchemaGenerator, self).create_view(callback, method, request)
        if is_disabled_action(view):
            view.exclude_from_schema = True

        return view
def has_permission(self, request, view):
        """Check list and create permissions based on sign and filters."""
        if view.suffix == 'Instance':
            return True

        filter_and_actions = self._get_filter_and_actions(
            request.query_params.get('sign'),
            view.action,
            '{}.{}'.format(
                view.queryset.model._meta.app_label,
                view.queryset.model._meta.model_name
            )
        )
        if not filter_and_actions:
            return False
        if request.method == 'POST':
            for key, value in request.data.iteritems():
                # Do unicode conversion because value will always be a
                # string
                if (key in filter_and_actions['filters'] and not
                        unicode(filter_and_actions['filters'][key]) == unicode(value)):
                    return False
        return True
def get_object_or_404(self, queryset, *filter_args, **filter_kwargs):
        """Return an object or raise a 404.

        Same as Django's standard shortcut, but make sure to raise 404
        if the filter_kwargs don't match the required types.
        """
        if isinstance(queryset, CachedQueryset):
            try:
                return queryset.get(*filter_args, **filter_kwargs)
            except queryset.model.DoesNotExist:
                raise Http404(
                    'No %s matches the given query.' % queryset.model)
        else:
            return get_object_or_404(queryset, *filter_args, **filter_kwargs)
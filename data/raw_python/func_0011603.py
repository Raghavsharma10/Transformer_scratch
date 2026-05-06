def get_object(self, queryset=None):
        """
        Return the object the view is displaying.

        Same as rest_framework.generics.GenericAPIView, but:
        - Failed assertions instead of deprecations
        """
        # Determine the base queryset to use.
        assert queryset is None, "Passing a queryset is disabled"
        queryset = self.filter_queryset(self.get_queryset())

        # Perform the lookup filtering.
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field
        lookup = self.kwargs.get(lookup_url_kwarg, None)
        assert lookup is not None, "Other lookup methods are disabled"
        filter_kwargs = {self.lookup_field: lookup}
        obj = self.get_object_or_404(queryset, **filter_kwargs)

        # May raise a permission denied
        self.check_object_permissions(self.request, obj)

        return obj
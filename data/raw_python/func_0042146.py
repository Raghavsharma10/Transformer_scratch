def get_queryset(self, **filter_kwargs):
        """
        Get the list of items for this view. This will
        call the `get_parent_object` method before doing
        anything else to ensure that a valid parent object
        is present. If a parent_object is returned it gets
        set to `self.parent_object`.

        If a queryset has been set then that queryset will be used.
        Otherwise the default manager for the provided
        model will be used.

        Once we have a queryset, the `get_filter` method
        is called and added to the queryset which is then
        returned.
        """
        self.parent_object = self.get_parent_object()

        if self.queryset is not None:
            queryset = self.queryset
            if hasattr(queryset, '_clone'):
                queryset = queryset._clone()
        elif self.model is not None:
            queryset = self.model._default_manager.filter()
        else:
            raise ImproperlyConfigured(u"'%s' must define 'queryset' or 'model'"
                                       % self.__class__.__name__)

        q_objects = self.get_filter(**filter_kwargs)
        queryset = queryset.filter()
        for q in q_objects:
            queryset = queryset.filter(q)

        return queryset
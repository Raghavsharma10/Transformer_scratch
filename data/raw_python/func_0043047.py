def get_object(self):
        """
        If a single object has been requested, will set
        `self.object` and return the object.
        """
        queryset = None
        slug = self.kwargs.get(self.slug_url_kwarg, None)

        if slug is not None:
            queryset = self.get_queryset()
            slug_field = self.slug_field
            queryset = queryset.filter(**{slug_field: slug})
            try:
                self.object = queryset.get()
            except ObjectDoesNotExist:
                raise http.Http404
        return self.object
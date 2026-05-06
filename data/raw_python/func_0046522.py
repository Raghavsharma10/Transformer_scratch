def get_object(self, queryset=None):
        """
        Returns the object the view is displaying.
        """
        if queryset is None:
            queryset = self.get_queryset()

        # Take a GET parameter instead of URLConf variable.
        try:
            pk = long(self.request.GET[self.pk_url_kwarg])
        except (KeyError, ValueError):
            raise Http404("Invalid Parameters")
        queryset = queryset.filter(pk=pk)

        try:
            # Get the single item from the filtered queryset
            obj = queryset.get()
        except ObjectDoesNotExist as e:
            raise Http404(e)
        return obj
def get_selected(self, request):
        """
        Returns a queryset of the selected objects as specified by \
        a GET or POST request.
        """
        obj = self.get_object()
        queryset = None
        # if single-object URL not used, check for selected objects
        if not obj:
            if request.GET.get(CHECKBOX_NAME):
                selected = request.GET.get(CHECKBOX_NAME).split(',')
            else:
                selected = request.POST.getlist(CHECKBOX_NAME)
        else:
            selected = [obj.pk]

        queryset = self.get_queryset().filter(pk__in=selected)
        return queryset
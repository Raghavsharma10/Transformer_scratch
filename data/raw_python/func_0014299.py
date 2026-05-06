def list(self, request):
        """
        List objects of a model. By default will show page 1 with 20 objects on it.

        **Usage**::

            params = {"items_per_page":10,"page":2} //all params are optional
            $.post("/ajax/{app}/{model}/list.json"),params)

        """

        max_items_per_page = getattr(self, 'max_per_page',
                                      getattr(settings, 'AJAX_MAX_PER_PAGE', 100))
        requested_items_per_page = request.POST.get("items_per_page", 20)
        items_per_page = min(max_items_per_page, requested_items_per_page)
        current_page = request.POST.get("current_page", 1)

        if not self.can_list(request.user):
            raise AJAXError(403, _("Access to this endpoint is forbidden"))

        objects = self.get_queryset(request)

        paginator = Paginator(objects, items_per_page)

        try:
            page = paginator.page(current_page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page.
            page = paginator.page(1)
        except EmptyPage:
            # If page is out of range (e.g. 9999), return empty list.
            page = EmptyPageResult()

        data = [encoder.encode(record) for record in page.object_list]
        return EnvelopedResponse(data=data, metadata={'total': paginator.count})
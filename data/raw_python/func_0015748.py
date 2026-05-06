def paginate_queryset(self, queryset, page_size):
        """
        Returns tuple containing paginator instance, page instance,
        object list, and whether there are other pages.

        :param queryset: the queryset instance to paginate.
        :param page_size: the number of instances per page.
        :rtype: tuple.
        """
        paginator = self.get_paginator(
            queryset,
            page_size,
            orphans                 = self.get_paginate_orphans(),
            allow_empty_first_page  = self.get_allow_empty()
        )

        page_kwarg  = self.page_kwarg
        #noinspection PyUnresolvedReferences
        page_num    = self.kwargs.get(page_kwarg) or self.request.GET.get(page_kwarg) or 1

        # Default to a valid page.
        try:
            page = paginator.page(page_num)
        except PageNotAnInteger:
            page = paginator.page(1)
        except EmptyPage:
            page = paginator.page(paginator.num_pages)

        #noinspection PyRedundantParentheses
        return (paginator, page, page.object_list, page.has_other_pages())
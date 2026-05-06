def get_list_data(self, request, **kwargs):
        """
        Returns the data needed for displaying the list.
        Returns a dictionary that should be treating as
        context with the following arguments:

        * **list** - The list of data to be displayed. This is \
        an instance of a wrapper class that combines the queryset \
        and formset and provides looping and labels and sorting controls.
        * **filter_form** - An instance of your filter form.
        * **is_paginated** - Is this list paginated, True or False.
        * **paginator** - The paginator object if available.
        * **page_obj** - A pagination object if available.
        * **show_form** - Should the rendering template show form controls.
        """

        self.object_list = self.get_queryset()
        self._verify_list()

        sort_field = None
        order_type = None
        if self.can_sort:
            default = None
            default_order = helpers.AdminList.ASC
            if self.object_list.ordered:
                if self.object_list.query.order_by:
                    default = self.object_list.query.order_by[0]
                else:
                    default = self.object_list.model._meta.ordering[0]
                if default.startswith('-'):
                    default = default[1:]
                    default_order = helpers.AdminList.DESC

            sort_field = request.GET.get('sf', default)
            order_type = request.GET.get('ot', default_order)

        queryset = self._sort_queryset(self.object_list, sort_field,
                                       order_type)

        if self.request.method == 'POST' and self.can_submit:
            formset = self.get_formset(data=self.request.POST, queryset=queryset)
            is_paginated, page, paginator, queryset = self._paginate_queryset(queryset)
        else:
            is_paginated, page, paginator, queryset = self._paginate_queryset(queryset)
            formset = self.get_formset(queryset=queryset)

        visible_fields = self.get_visible_fields(formset)
        adm_list = helpers.AdminList(formset, queryset, visible_fields,
                                       sort_field, order_type,
                                       self.model_name)

        actions = self.get_action_context(request)
        data = {
            'list': adm_list,
            'filter_form': self.get_filter_form(),
            'page_obj': page,
            'is_paginated': is_paginated,
            'show_form': (self.can_submit and formset is not None),
            'paginator': paginator,
            'checkbox_name' : CHECKBOX_NAME,
            'actions' : actions,
        }

        return data
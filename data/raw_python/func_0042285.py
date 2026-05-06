def render(self, request, **kwargs):
        """
        Returns a JSON representation of a objects list page.
        The json has the following attributes:

        * **is_paginated** - Is the list paginated.
        * **results** - A list of objects, where each object has an \
        attribute/value for each field in the list. An 'id' attribute \
        is always included.
        * **fields** - An object who's properties are the fields \
        in the results list. Each property will have an object with \
        the the following attributes:
            * **name** - The verbose name of the field.
            * **sortable** - Can this column be sorted. True or False.
            * **order_type** - What is the current order of this column.

        The following attributes only appear if the list is paginated:

        * **count** - If the list is paginated, how many objects \
        total are there.
        * **page** - Current page number.
        * **next** - The full link to the next page.
        * **previous** - The full link to the previous page.

        If the list can be filtered the following attribute is included:

        * **params** - An object who's properties are the filter options. \
            Each property contains an object with the following attributes:
            * **value** - If the current result list has been filtered by \
            this field then value will contain the filter value that was used.
            * **choices** - If the field is a choice field this will contain \
            the options.

        Example JSON:

        ::

            {"count": 1,
            "fields": {
                "name": {"sortable": true, "name": "name", "order_type": "asc"}
            },
            "results": [{"id": 12, "name": "Test"}],
            "next": "",
            "params": {"name": {"value": null}},
            "is_paginated": true,
            "page": 1,
            "previous": ""}
        """
        data = {
            'is_paginated': kwargs.get('is_paginated')
        }

        if data.get('is_paginated'):
            page = kwargs['page_obj']

            next_p = ''
            previous = ''
            if page.has_next():
                next_p = self.get_different_page(request, page.number + 1)

            if page.has_previous():
                previous = self.get_different_page(request, page.number - 1)

            data.update({
                'count': page.paginator.count,
                'page': page.number,
                'next': next_p,
                'previous': previous,
            })

        if kwargs.get('filter_form'):
            exclude = request.GET.getlist('exclude')
            filter_form = {}
            form = kwargs.get('filter_form')
            for name in form.get_search_fields(exclude):
                k = form[name]
                obj = {}
                obj['value'] = k.value()
                obj['label'] = k.label
                if hasattr(k.field, 'choices'):
                    obj['choices'] = k.field.choices

                filter_form[k.name] = obj

            data['params'] = filter_form

        adm_list = kwargs['list']
        data['fields'] = self.get_fields(adm_list)
        data['results'] = self.get_object_list(adm_list)
        return http.HttpResponse(json.dumps(data, cls=DjangoJSONEncoder))
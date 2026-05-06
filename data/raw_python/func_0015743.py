def get_context_data(self, **kwargs):
        """
        Returns context dictionary for view.

        :rtype: dict.
        """
        #noinspection PyUnresolvedReferences
        query_str           = self.request.GET.get('q', None)
        queryset            = kwargs.pop('object_list', self.object_list)
        context_object_name = self.get_context_object_name(queryset)

        # Build the context dictionary.
        context = {
            'ordering':     self.get_ordering(),
            'query_string': query_str,
            'is_searching': bool(query_str),
        }

        # Add extra variables to context for non-AJAX requests.
        #noinspection PyUnresolvedReferences
        if not self.request.is_ajax() or kwargs.get('force_search', False):
            context.update({
                'search_form':  self.get_search_form(),
                'popular_tags': self.model.popular_tags()
            })

        if context_object_name is not None:
            context[context_object_name] = queryset

        # Update context with any additional keyword arguments.
        context.update(kwargs)

        return super(IndexView, self).get_context_data(**context)
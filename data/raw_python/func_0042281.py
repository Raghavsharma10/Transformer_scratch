def update_kwargs(self, request, **kwargs):
        """
        Hook for adding data to the context before
        rendering a template.

        :param kwargs: The current context keyword arguments.
        :param request: The current request object.
        """
        if not 'base' in kwargs:
            kwargs['base'] = self.base
            if request.is_ajax() or request.GET.get('json'):
                kwargs['base'] = self.partial_base

        return kwargs
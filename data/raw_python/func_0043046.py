def get_context_data(self, **kwargs):
        """
        Hook for adding arguments to the context.
        """

        context = {'obj': self.object }
        if 'queryset' in kwargs:
            context['conf_msg'] = self.get_confirmation_message(kwargs['queryset'])
        context.update(kwargs)
        return context
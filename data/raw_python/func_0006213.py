def get(self, request, *args, **kwargs):
        """
        Django view get function.

        Add items of extra_context, crumbs and grid to context.

        Args:
            request (): Django's request object.
            *args (): request args.
            **kwargs (): request kwargs.

        Returns:
            response: render to response with context.
        """
        context = self.get_context_data(**kwargs)
        context.update(self.extra_context)
        context['crumbs'] = self.get_crumbs()
        context['title'] = self.title
        context['suit'] = 'suit' in settings.INSTALLED_APPS
        if context.get('dashboard_grid', None) is None and self.grid:
            context['dashboard_grid'] = self.grid
        return self.render_to_response(context)
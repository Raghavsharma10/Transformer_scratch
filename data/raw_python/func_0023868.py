def get_context_data(self, **kwargs):
        """Allow adding a 'render_description' parameter"""
        context = super(ScheduleXmlView, self).get_context_data(**kwargs)
        if self.request.GET.get('render_description', None) == '1':
            context['render_description'] = True
        else:
            context['render_description'] = False
        return context
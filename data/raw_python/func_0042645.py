def render(self, request, **kwargs):
        """
        Renders this view. Adds cancel_url to the context.
        If the request get parameters contains 'popup' then
        the `render_type` is set to 'popup'.
        """
        if request.GET.get('popup'):
            self.render_type = 'popup'
            kwargs['popup'] = 1

        kwargs['cancel_url'] = self.get_cancel_url()
        if not self.object:
            kwargs['single_title'] = True
        return super(FormView, self).render(request, **kwargs)
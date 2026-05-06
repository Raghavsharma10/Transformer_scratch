def set_renderers(self, request=None, context=None, template_name=None, early=False):
        """
        Makes sure that the renderers attribute on the request is up
        to date. renderers_for_view keeps track of the view that
        is attempting to render the request, so that if the request
        has been delegated to another view we know to recalculate
        the applicable renderers. When called multiple times on the
        same view this will be very low-cost for subsequent calls.
        """
        request, context, template_name = self.get_render_params(request, context, template_name)

        args = (self.conneg, context, template_name,
                self._default_format, self._force_fallback_format, self._format_override_parameter)
        if getattr(request, 'renderers_for_args', None) != args:
            fallback_formats = self._force_fallback_format or ()
            if not isinstance(fallback_formats, (list, tuple)):
                fallback_formats = (fallback_formats,)
            request.renderers = self.conneg.get_renderers(request=request,
                                                          context=context,
                                                          template_name=template_name,
                                                          accept_header=request.META.get('HTTP_ACCEPT'),
                                                          formats=self.format_override,
                                                          default_format=self._default_format,
                                                          fallback_formats=fallback_formats,
                                                          early=early)
            request.renderers_for_view = args
        if self._include_renderer_details_in_context:
            self.context['renderers'] = [self.renderer_for_context(request, r) for r in self.conneg.renderers]
        return request.renderers
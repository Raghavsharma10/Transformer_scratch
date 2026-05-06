def render_to_string(self, request, template, context, content_instance=None):
        """
        Render a custom template with the :class:`~PluginContext` as context instance.
        """
        if not content_instance:
            content_instance = PluginContext(request)

        content_instance.update(context)
        return render_to_string(template, content_instance.flatten(), request=request)
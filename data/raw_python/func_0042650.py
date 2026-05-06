def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests.
        Sets the renderer to be a RenderResponse instance
        that uses `default_template` as the template.

        The `preview_view` callable is called and passed to `render`
        method as the data keyword argument.
        """

        self.renders = {
            'response': renders.RenderResponse(template=self.default_template),
        }
        kwargs = self.get_preview_kwargs(**kwargs)
        view = self.preview_view.as_string()
        return self.render(request, data=view(request, **kwargs))
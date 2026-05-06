def render(self, request, collect_render_data=True, **kwargs):
        """
        Render this view. This will call the render method
        on the render class specified.

        :param request: The request object
        :param collect_render_data: If True we will call \
        the get_render_data method to pass a complete context \
        to the renderer.
        :param kwargs: Any other keyword arguments that should \
        be passed to the renderer.
        """
        assert self.render_type in self.renders
        render = self.renders[self.render_type]
        if collect_render_data:
            kwargs = self.get_render_data(**kwargs)

        return render.render(request, **kwargs)
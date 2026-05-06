def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests. If there is
        a GET parameter type=choice, then the render_type
        will be set to 'choices' to return a JSON version
        of this list. Calls `render` with the data from the
        `get_list_data` method as context.
        """

        if request.GET.get('type') == 'choices':
            self.render_type = 'choices'
            self.can_submit = False

        data = self.get_list_data(request, **kwargs)
        return self.render(request, **data)
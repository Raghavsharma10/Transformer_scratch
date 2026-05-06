def post(self, request, *args, **kwargs):
        """
        Method for handling POST requests.
        Checks for a modify confirmation and performs
        the action by calling `process_action`.

        """
        queryset = self.get_selected(request)

        if request.POST.get('modify'):
            response = self.process_action(request, queryset)
            if not response:
                url = self.get_done_url()
                return self.render(request, redirect_url=url)
            else:
                return response
        else:
            return self.render(request, redirect_url=request.build_absolute_uri())
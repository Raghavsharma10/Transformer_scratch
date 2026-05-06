def success_response(self, message=None):
        """
        Returns a 'render redirect' to the result of the
        `get_success_url` method.
        """

        return self.render(self.request,
                           redirect_url=self.get_success_url(),
                           obj=self.object,
                           message=message,
                           collect_render_data=False)
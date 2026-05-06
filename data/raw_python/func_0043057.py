def get_done_url(self):
        """
        Returns the url to redirect to after a successful update.
        The get_view_url will be called on the current bundle using
        `self.redirect_to_view` as the view name.
        """
        url = self.bundle.get_view_url(self.redirect_to_view,
                                        self.request.user, {}, self.kwargs)
        return self.customized_return_url(url)
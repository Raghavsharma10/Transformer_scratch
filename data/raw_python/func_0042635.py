def get_cancel_url(self):
        """
        Returns the cancel url for this view.

        if `self.cancel_view` is None the current url will
        be used. Otherwise the get_view_url will be called with
        the current bundle using `self.cancel_view` as the
        view name.
        """
        if self.cancel_view:
            url = self.bundle.get_view_url(self.cancel_view,
                                            self.request.user, {},
                                            self.kwargs)
        else:
            url = self.request.build_absolute_uri()

        return self.customized_return_url(url)
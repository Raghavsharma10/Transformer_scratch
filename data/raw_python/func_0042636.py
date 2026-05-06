def get_success_url(self):
        """
        Returns the url to redirect to after a successful update.

        if `self.redirect_to_view` is None the current url will
        be used. Otherwise the get_view_url will be called
        on the current bundle using `self.redirect_to_view` as the
        view name. If the name is "main" or "main_list" no object
        will be passed. Otherwise `self.object` will be passed as
        a kwarg.
        """

        if self.redirect_to_view:
            kwargs = {}
            if self.redirect_to_view != 'main' and \
                        self.redirect_to_view != 'main_list':
                kwargs['object'] = self.object
            return self.bundle.get_view_url(self.redirect_to_view,
                                            self.request.user, kwargs,
                                            self.kwargs)
        else:
            return self.request.build_absolute_uri()
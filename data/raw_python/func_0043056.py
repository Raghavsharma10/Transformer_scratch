def get_object_url(self):
        """
        Returns the url to link to the object
        The get_view_url will be called on the current bundle using
        'edit` as the view name.
        """
        return self.bundle.get_view_url('edit',
                                        self.request.user, {}, self.kwargs)
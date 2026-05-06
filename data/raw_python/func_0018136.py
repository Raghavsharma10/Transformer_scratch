def unpublish_view(self, request, object_id):
        """
        Instantiates a class-based view that redirects to Wagtail's 'unpublish'
        view for models that extend 'Page' (if the user has sufficient
        permissions). We do this via our own view so that we can reliably
        control redirection of the user back to the index_view once the action
        is completed. The view class used can be overridden by changing the
        'unpublish_view_class' attribute.
        """
        kwargs = {'model_admin': self, 'object_id': object_id}
        view_class = self.unpublish_view_class
        return view_class.as_view(**kwargs)(request)
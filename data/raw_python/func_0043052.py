def get_object(self):
        """
        Get the object for publishing
        Raises a http404 error if the object is not found.
        """
        obj = super(PublishActionView, self).get_object()

        if obj:
            if not hasattr(obj, 'publish'):
                raise http.Http404

        return obj
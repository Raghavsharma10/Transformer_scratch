def get_object(self):
        """
        Get the object for publishing
        Raises a http404 error if the object is not found.
        """
        obj = super(PublishView, self).get_object()

        if not obj or not hasattr(obj, 'publish'):
            raise http.Http404

        return obj
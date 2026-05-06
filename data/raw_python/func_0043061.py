def get_object(self):
        """
        Get the object for previewing.
        Raises a http404 error if the object is not found.
        """
        obj = super(DeleteView, self).get_object()

        if not obj:
            raise http.Http404

        return obj
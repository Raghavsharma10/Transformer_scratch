def dispatch(self, request, *args, **kwargs):
        """Adds useful objects to the class and performs security checks."""
        self._add_next_and_user(request)
        self.content_object = None
        self.content_type = None
        self.object_id = kwargs.get('object_id', None)

        if kwargs.get('content_type'):
            # Check if the user forged the URL and posted a non existant
            # content type
            try:
                self.content_type = ContentType.objects.get(
                    model=kwargs.get('content_type'))
            except ContentType.DoesNotExist:
                raise Http404

        if self.content_type:
            # Check if the user forged the URL and tries to append the image to
            # an object that does not exist
            try:
                self.content_object = \
                    self.content_type.get_object_for_this_type(
                        pk=self.object_id)
            except ObjectDoesNotExist:
                raise Http404

        if self.content_object and hasattr(self.content_object, 'user'):
            # Check if the user forged the URL and tries to append the image to
            # an object that does not belong to him
            if not self.content_object.user == self.user:
                raise Http404

        return super(CreateImageView, self).dispatch(request, *args, **kwargs)
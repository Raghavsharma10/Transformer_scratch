def _get_method_kwargs(self):
        """
        Helper method. Returns kwargs needed to filter the correct object.

        Can also be used to create the correct object.

        """
        method_kwargs = {
            'user': self.user,
            'content_type': self.ctype,
            'object_id': self.content_object.pk,
        }
        return method_kwargs
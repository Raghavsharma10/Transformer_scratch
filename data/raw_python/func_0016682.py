def get_success_url(self):
        """
        Returns the success URL.

        This is either the given `next` URL parameter or the content object's
        `get_absolute_url` method's return value.

        """
        if self.next:
            return self.next
        if self.object and self.object.content_object:
            return self.object.content_object.get_absolute_url()
        raise Exception(
            'No content object given. Please provide ``next`` in your POST'
            ' data')
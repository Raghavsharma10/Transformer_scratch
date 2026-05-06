def get_object_url(self):
        """
        Returns the url where this object can be edited.
        """

        if self.kwargs.get(self.slug_url_kwarg, False) == \
                    unicode(getattr(self.object, self.slug_field, "")) \
                    and not self.force_add:
            url = self.request.build_absolute_uri()
        else:
            url = self.bundle.get_view_url('edit', self.request.user,
                                           {'object': self.object},
                                           self.kwargs)
        return url
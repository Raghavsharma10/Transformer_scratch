def get_item(self, **kwargs):
        """ Get collection item taking into account generated queryset
        of parent view.

        This method allows working with nested resources properly. Thus an item
        returned by this method will belong to its parent view's queryset, thus
        filtering out objects that don't belong to the parent object.

        Returns an object from the applicable ACL. If ACL wasn't applied, it is
        applied explicitly.
        """
        if six.callable(self.context):
            self.reload_context(es_based=False, **kwargs)

        objects = self._parent_queryset()
        if objects is not None and self.context not in objects:
            raise JHTTPNotFound('{}({}) not found'.format(
                self.Model.__name__,
                self._get_context_key(**kwargs)))

        return self.context
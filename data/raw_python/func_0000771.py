def get_collection(self, **kwargs):
        """ Get objects collection taking into account generated queryset
        of parent view.

        This method allows working with nested resources properly. Thus a
        queryset returned by this method will be a subset of its parent
        view's queryset, thus filtering out objects that don't belong to
        the parent object.
        """
        self._query_params.update(kwargs)
        objects = self._parent_queryset()
        if objects is not None:
            return self.Model.filter_objects(
                objects, **self._query_params)
        return self.Model.get_collection(**self._query_params)
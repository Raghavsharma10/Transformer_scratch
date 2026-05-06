def get_collection_es(self):
        """ Get ES objects collection taking into account the generated
        queryset of parent view.

        This method allows working with nested resources properly. Thus a
        queryset returned by this method will be a subset of its parent view's
        queryset, thus filtering out objects that don't belong to the parent
        object.
        """
        objects_ids = self._parent_queryset_es()

        if objects_ids is not None:
            objects_ids = self.get_es_object_ids(objects_ids)
            if not objects_ids:
                return []
            self._query_params['id'] = objects_ids

        return super(ESBaseView, self).get_collection_es()
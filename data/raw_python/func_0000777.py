def get_item_es(self, **kwargs):
        """ Get ES collection item taking into account generated queryset
        of parent view.

        This method allows working with nested resources properly. Thus an item
        returned by this method will belong to its parent view's queryset, thus
        filtering out objects that don't belong to the parent object.

        Returns an object retrieved from the applicable ACL. If an ACL wasn't
        applied, it is applied explicitly.
        """
        item_id = self._get_context_key(**kwargs)
        objects_ids = self._parent_queryset_es()
        if objects_ids is not None:
            objects_ids = self.get_es_object_ids(objects_ids)

        if six.callable(self.context):
            self.reload_context(es_based=True, **kwargs)

        if (objects_ids is not None) and (item_id not in objects_ids):
            raise JHTTPNotFound('{}(id={}) resource not found'.format(
                self.Model.__name__, item_id))

        return self.context
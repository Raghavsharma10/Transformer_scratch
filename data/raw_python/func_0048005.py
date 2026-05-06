def get_item(self, item_id):
        """Gets the ``Item`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Item`` may have a different
        ``Id`` than requested, such as the case where a duplicate ``Id``
        was assigned to an ``Item`` and retained for compatibility.

        arg:    item_id (osid.id.Id): the ``Id`` of the ``Item`` to
                retrieve
        return: (osid.assessment.Item) - the returned ``Item``
        raise:  NotFound - no ``Item`` found with the given ``Id``
        raise:  NullArgument - ``item_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(item_id, 'assessment').get_identifier())},
                 **self._view_filter()))
        return objects.Item(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)
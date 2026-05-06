def get_items_by_genus_type(self, item_genus_type):
        """Gets an ``ItemList`` corresponding to the given assessment item genus ``Type`` which does not include assessment items of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known assessment
        items or an error results. Otherwise, the returned list may
        contain only those assessment items that are accessible through
        this session.

        arg:    item_genus_type (osid.type.Type): an assessment item
                genus type
        return: (osid.assessment.ItemList) - the returned ``Item`` list
        raise:  NullArgument - ``item_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('assessment',
                                         collection='Item',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(item_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.ItemList(result, runtime=self._runtime, proxy=self._proxy)
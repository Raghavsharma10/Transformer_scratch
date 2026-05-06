def get_resources_by_ids(self, resource_ids):
        """Gets a ``ResourceList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the resources
        specified in the ``Id`` list, in the order of the list,
        including duplicates, or an error results if an ``Id`` in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible ``Resources`` may be omitted from the list and may
        present the elements in any order including returning a unique
        set.

        arg:    resource_ids (osid.id.IdList): the list of ``Ids`` to
                retrieve
        return: (osid.resource.ResourceList) - the returned ``Resource``
                list
        raise:  NotFound - an ``Id was`` not found
        raise:  NullArgument - ``resource_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_ids
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        object_id_list = []
        for i in resource_ids:
            object_id_list.append(ObjectId(self._get_id(i, 'resource').get_identifier()))
        result = collection.find(
            dict({'_id': {'$in': object_id_list}},
                 **self._view_filter()))
        result = list(result)
        sorted_result = []
        for object_id in object_id_list:
            for object_map in result:
                if object_map['_id'] == object_id:
                    sorted_result.append(object_map)
                    break
        return objects.ResourceList(sorted_result, runtime=self._runtime, proxy=self._proxy)
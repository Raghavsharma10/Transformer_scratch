def get_item_ids_by_bank(self, bank_id):
        """Gets the list of ``Item``  ``Ids`` associated with a ``Bank``.

        arg:    bank_id (osid.id.Id): ``Id`` of the ``Bank``
        return: (osid.id.IdList) - list of related item ``Ids``
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for item in self.get_items_by_bank(bank_id):
            id_list.append(item.get_id())
        return IdList(id_list)
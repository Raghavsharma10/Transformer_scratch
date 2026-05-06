def get_bank_ids_by_item(self, item_id):
        """Gets the list of ``Bank``  ``Ids`` mapped to an ``Item``.

        arg:    item_id (osid.id.Id): ``Id`` of an ``Item``
        return: (osid.id.IdList) - list of bank ``Ids``
        raise:  NotFound - ``item_id`` is not found
        raise:  NullArgument - ``item_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - assessment failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_item_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        item = lookup_session.get_item(item_id)
        id_list = []
        for idstr in item._my_map['assignedBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)
def get_items_by_bank(self, bank_id):
        """Gets the list of ``Items`` associated with a ``Bank``.

        arg:    bank_id (osid.id.Id): ``Id`` of the ``Bank``
        return: (osid.assessment.ItemList) - list of related items
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_item_lookup_session_for_bank(bank_id, proxy=self._proxy)
        lookup_session.use_isolated_bank_view()
        return lookup_session.get_items()
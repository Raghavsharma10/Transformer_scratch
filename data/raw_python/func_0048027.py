def get_items_by_banks(self, bank_ids):
        """Gets the list of ``Items`` corresponding to a list of ``Banks``.

        arg:    bank_ids (osid.id.IdList): list of bank ``Ids``
        return: (osid.assessment.ItemList) - list of items
        raise:  NullArgument - ``bank_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - assessment failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        item_list = []
        for bank_id in bank_ids:
            item_list += list(
                self.get_items_by_bank(bank_id))
        return objects.ItemList(item_list)
def get_assignable_bank_ids(self, bank_id):
        """Gets a list of bank including and under the given bank node in which any assessment part can be assigned.

        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        return: (osid.id.IdList) - list of assignable bank ``Ids``
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # This will likely be overridden by an authorization adapter
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        banks = lookup_session.get_banks()
        id_list = []
        for bank in banks:
            id_list.append(bank.get_id())
        return IdList(id_list)
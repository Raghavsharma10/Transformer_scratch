def get_objective_banks_by_ids(self, objective_bank_ids=None):
        """Gets a ObjectiveBankList corresponding to the given IdList.
        In plenary mode, the returned list contains all of the objective
        banks specified in the Id list, in the order of the list,
        including duplicates, or an error results if an Id in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible ObjectiveBank objects may be omitted from the list
        and may present the elements in any order including returning a
        unique set.
        arg:    objectiveBankIds (osid.id.IdList): the list of Ids to
                retrieve
        return: (osid.learning.ObjectiveBankList) - the returned
                ObjectiveBank list
        raise:  NotFound - an Id was not found
        raise:  NullArgument - objectiveBankIds is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_bank_ids is None:
            raise NullArgument()
        banks = []
        # The following runs really slow. Perhaps get all banks and then inspect result for ids
        for i in objective_bank_ids:
            bank = None
            url_path = construct_url('objective_banks',
                                     bank_id=i)
            try:
                bank = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._objective_bank_view == PLENARY:
                    raise
                else:
                    pass
            if bank:
                if not (self._objective_bank_view == COMPARATIVE and
                        bank in banks):
                    banks.append(bank)
        return objects.ObjectiveBankList(banks)
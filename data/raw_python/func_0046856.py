def delete_objective_bank(self, objective_bank_id=None):
        """Deletes an ObjectiveBank.

        arg:    objectiveBankId (osid.id.Id): the Id of the
                ObjectiveBank to remove
        raise:  NotFound - objectiveBankId not found
        raise:  NullArgument - objectiveBankId is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        if objective_bank_id is None:
            raise NullArgument()
        if not isinstance(objective_bank_id, ABCId):
            raise InvalidArgument('argument type is not an osid Id')

        # Check for "sandbox" genus type.  Hardcoded for now:
        try:
            objective_bank = ObjectiveBankLookupSession(proxy=self._proxy,
                                                        runtime=self._runtime).get_objective_bank(objective_bank_id)
        except Exception:
            raise
        # if objective_bank._my_map['genusTypeId'] != 'mc3-objectivebank%3Amc3.learning.objectivebank.sandbox%40MIT-OEIT':
        #     raise PermissionDenied('Handcar only supports deleting \'sandbox\' type ObjectiveBanks')

        url_path = construct_url('objective_banks',
                                 bank_id=objective_bank_id)
        result = self._delete_request(url_path)
        return objects.ObjectiveBank(result)
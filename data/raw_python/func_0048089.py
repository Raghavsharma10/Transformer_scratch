def get_bank_ids_by_assessment_taken(self, assessment_taken_id):
        """Gets the list of ``Bank``  ``Ids`` mapped to an ``AssessmentTaken``.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of an
                ``AssessmentTaken``
        return: (osid.id.IdList) - list of bank ``Ids``
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_assessment_taken_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        assessment_taken = lookup_session.get_assessment_taken(assessment_taken_id)
        id_list = []
        for idstr in assessment_taken._my_map['assignedBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)
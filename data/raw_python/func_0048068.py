def get_bank_ids_by_assessment_offered(self, assessment_offered_id):
        """Gets the list of ``Bank``  ``Ids`` mapped to an ``AssessmentOffered``.

        arg:    assessment_offered_id (osid.id.Id): ``Id`` of an
                ``AssessmentOffered``
        return: (osid.id.IdList) - list of bank ``Ids``
        raise:  NotFound - ``assessment_offered_id`` is not found
        raise:  NullArgument - ``assessment_offered_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_assessment_offered_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        assessment_offered = lookup_session.get_assessment_offered(assessment_offered_id)
        id_list = []
        for idstr in assessment_offered._my_map['assignedBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)
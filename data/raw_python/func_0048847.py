def get_bank_ids_by_assessment_part(self, assessment_part_id):
        """Gets the ``Bank``  ``Ids`` mapped to an ``AssessmentPart``.

        arg:    assessment_part_id (osid.id.Id): ``Id`` of an
                ``AssessmentPart``
        return: (osid.id.IdList) - list of banks
        raise:  NotFound - ``assessment_part_id`` is not found
        raise:  NullArgument - ``assessment_part_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
        lookup_session = mgr.get_assessment_part_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_bank_view()
        assessment_part = lookup_session.get_assessment_part(assessment_part_id)
        id_list = []
        for idstr in assessment_part._my_map['assignedBankIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)
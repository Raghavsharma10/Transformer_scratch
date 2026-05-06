def get_assessment_parts_by_bank(self, bank_id):
        """Gets the list of assessment parts associated with an ``Bank``.

        arg:    bank_id (osid.id.Id): ``Id`` of the ``Bank``
        return: (osid.assessment.authoring.AssessmentPartList) - list of
                related assessment parts
        raise:  NotFound - ``bank_id`` is not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('ASSESSMENT_AUTHORING', local=True)
        lookup_session = mgr.get_assessment_part_lookup_session_for_bank(bank_id, proxy=self._proxy)
        lookup_session.use_isolated_bank_view()
        return lookup_session.get_assessment_parts()
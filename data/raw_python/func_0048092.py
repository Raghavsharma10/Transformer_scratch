def unassign_assessment_taken_from_bank(self, assessment_taken_id, bank_id):
        """Removes an ``AssessmentTaken`` from a ``Bank``.

        arg:    assessment_taken_id (osid.id.Id): the ``Id`` of the
                ``AssessmentTaken``
        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        raise:  NotFound - ``assessment_taken_id`` or ``bank_id`` not
                found or ``assessment_taken_id`` not assigned to
                ``bank_id``
        raise:  NullArgument - ``assessment_taken_id`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_bank(bank_id)  # to raise NotFound
        self._unassign_object_from_catalog(assessment_taken_id, bank_id)
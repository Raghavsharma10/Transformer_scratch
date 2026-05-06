def unassign_assessment_part_from_bank(self, assessment_part_id, bank_id):
        """Removes an ``AssessmentPart`` from an ``Bank``.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        raise:  NotFound - ``assessment_part_id`` or ``bank_id`` not
                found or ``assessment_part_id`` not assigned to
                ``bank_id``
        raise:  NullArgument - ``assessment_part_id`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_bank(bank_id)  # to raise NotFound
        self._unassign_object_from_catalog(assessment_part_id, bank_id)
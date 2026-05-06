def assign_assessment_part_to_bank(self, assessment_part_id, bank_id):
        """Adds an existing ``AssessmentPart`` to an ``Bank``.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        raise:  AlreadyExists - ``assessment_part_id`` is already
                assigned to ``bank_id``
        raise:  NotFound - ``assessment_part_id`` or ``bank_id`` not
                found
        raise:  NullArgument - ``assessment_part_id`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_bank(bank_id)  # to raise NotFound
        self._assign_object_to_catalog(assessment_part_id, bank_id)
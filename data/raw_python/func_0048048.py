def assign_assessment_to_bank(self, assessment_id, bank_id):
        """Adds an existing ``Assessment`` to a ``Bank``.

        arg:    assessment_id (osid.id.Id): the ``Id`` of the
                ``Assessment``
        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        raise:  AlreadyExists - ``assessment_id`` is already assigned to
                ``bank_id``
        raise:  NotFound - ``assessment_id`` or ``bank_id`` not found
        raise:  NullArgument - ``assessment_id`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.assign_resource_to_bin
        mgr = self._get_provider_manager('ASSESSMENT', local=True)
        lookup_session = mgr.get_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_bank(bank_id)  # to raise NotFound
        self._assign_object_to_catalog(assessment_id, bank_id)
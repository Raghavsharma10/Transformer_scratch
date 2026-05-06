def get_assessment_part_item_design_session_for_bank(self, bank_id, proxy):
        """Gets the ``OsidSession`` associated with the assessment part item design service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the ``Bank``
        return: (osid.assessment.authoring.AssessmentPartItemDesignSession)
                - an ``AssessmentPartItemDesignSession``
        raise:  NotFound - no ``Bank`` found by the given ``Id``
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_item_design()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_item_design()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_part_lookup():  # This is kludgy, but only until Tom fixes spec
            raise errors.Unimplemented()

        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        # pylint: disable=no-member
        return sessions.AssessmentPartItemDesignSession(bank_id, proxy=proxy, runtime=self._runtime)
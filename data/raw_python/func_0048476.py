def get_assessment_part_bank_assignment_session(self):
        """Gets the ``OsidSession`` associated with assigning assessment part to bank.

        return:
                (osid.assessment.authoring.AssessmentPartBankAssignmentS
                ession) - an ``AssessmentPartBankAssignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_assessment_part_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_bank_assignment()`` is ``true``.*

        """
        if not self.supports_assessment_part_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartBankAssignmentSession(runtime=self._runtime)
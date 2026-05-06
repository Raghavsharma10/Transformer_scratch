def get_assessment_part_bank_session(self):
        """Gets the ``OsidSession`` to lookup assessment part/bank mappings for assessment parts.

        return: (osid.assessment.authoring.AssessmentPartBankSession) -
                an ``AssessmentPartBankSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_bank()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_bank()`` is ``true``.*

        """
        if not self.supports_assessment_part_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartBankSession(runtime=self._runtime)
def get_assessment_part_item_session(self, proxy):
        """Gets the ``OsidSession`` associated with the assessment part item service.

        return: (osid.assessment.authoring.AssessmentPartItemSession)
                - an ``AssessmentPartItemSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_part_item()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_part_lookup()`` is ``true``.*

        """
        if not self.supports_assessment_part_lookup():  # This is kludgy, but only until Tom fixes spec
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssessmentPartItemSession(proxy=proxy, runtime=self._runtime)
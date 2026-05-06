def get_assessment_part_item_session(self, *args, **kwargs):
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
        if self._proxy_in_args(*args, **kwargs):
            raise errors.InvalidArgument('A Proxy object was received but not expected.')
        # pylint: disable=no-member
        return sessions.AssessmentPartItemSession(runtime=self._runtime)
def get_assessment_notification_session_for_bank(self, assessment_receiver, bank_id):
        """Gets the ``OsidSession`` associated with the assessment notification service for the given bank.

        arg:    assessment_receiver
                (osid.assessment.AssessmentReceiver): the assessment
                receiver interface
        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        return: (osid.assessment.AssessmentNotificationSession) - ``an
                _assessment_notification_session``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``assessment_receiver`` or ``bank_id`` is
                ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_assessment_notification()``
                or ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_notification()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_assessment_notification():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ItemNotificationSession(bank_id, runtime=self._runtime, receiver=assessment_receiver)
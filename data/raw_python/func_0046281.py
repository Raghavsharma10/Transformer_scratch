def get_assessment_notification_session(self, assessment_receiver):
        """Gets the notification session for notifications pertaining to assessment changes.

        arg:    assessment_receiver
                (osid.assessment.AssessmentReceiver): the assessment
                receiver interface
        return: (osid.assessment.AssessmentNotificationSession) - an
                ``AssessmentNotificationSession``
        raise:  NullArgument - ``assessment_receiver`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_assessment_notification()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_assessment_notification()`` is ``true``.*

        """
        if not self.supports_assessment_notification():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemNotificationSession(runtime=self._runtime, receiver=assessment_receiver)
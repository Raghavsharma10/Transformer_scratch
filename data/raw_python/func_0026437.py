def _send_invitation(self, enrollment, event):
        """Send an invitation mail to an open enrolment"""

        self.log('Sending enrollment status mail to user')

        self._send_mail(self.config.invitation_subject, self.config.invitation_mail, enrollment, event)
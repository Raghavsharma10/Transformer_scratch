def _send_acceptance(self, enrollment, password, event):
        """Send an acceptance mail to an open enrolment"""

        self.log('Sending acceptance status mail to user')

        if password is not "":
            password_hint = '\n\nPS: Your new password is ' + password + ' - please change it after your first login!'

            acceptance_text = self.config.acceptance_mail + password_hint
        else:
            acceptance_text = self.config.acceptance_mail

        self._send_mail(self.config.acceptance_subject, acceptance_text, enrollment, event)
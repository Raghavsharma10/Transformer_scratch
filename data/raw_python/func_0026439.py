def _send_mail(self, subject, template, enrollment, event):
        """Connect to mail server and send actual email"""

        context = {
            'name': enrollment.name,
            'invitation_url': self.invitation_url,
            'node_name': self.node_name,
            'node_url': self.node_url,
            'uuid': enrollment.uuid
        }

        mail = render(template, context)
        self.log('Mail:', mail, lvl=verbose)
        mime_mail = MIMEText(mail)
        mime_mail['Subject'] = render(subject, context)
        mime_mail['From'] = render(self.config.mail_from, {'hostname': self.hostname})
        mime_mail['To'] = enrollment.email

        self.log('MimeMail:', mime_mail, lvl=verbose)
        if self.config.mail_send is True:
            self.log('Sending mail to', enrollment.email)

            self.fireEvent(task(send_mail_worker, self.config, mime_mail, event), "enrolworkers")
        else:
            self.log('Not sending mail, here it is for debugging info:', mail, pretty=True)
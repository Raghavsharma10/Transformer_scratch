def email(self, subject, text_body, html_body=None, sender=None, **kwargs):
        # type: (str, str, Optional[str], Optional[str], Any) -> None
        """Emails a user.

        Args:
            subject (str): Email subject
            text_body (str): Plain text email body
            html_body (str): HTML email body
            sender (Optional[str]): Email sender. Defaults to SMTP username.
            **kwargs: See below
            mail_options (List): Mail options (see smtplib documentation)
            rcpt_options (List): Recipient options (see smtplib documentation)

        Returns:
            None
        """
        self.configuration.emailer().send([self.data['email']], subject, text_body, html_body=html_body, sender=sender,
                                          **kwargs)
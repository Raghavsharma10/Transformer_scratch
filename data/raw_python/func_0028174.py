def email_users(users, subject, text_body, html_body=None, sender=None, configuration=None, **kwargs):
        # type: (List['User'], str, str, Optional[str], Optional[str], Optional[Configuration], Any) -> None
        """Email a list of users

        Args:
            users (List[User]): List of users
            subject (str): Email subject
            text_body (str): Plain text email body
            html_body (str): HTML email body
            sender (Optional[str]): Email sender. Defaults to SMTP username.
            configuration (Optional[Configuration]): HDX configuration. Defaults to configuration of first user in list.
            **kwargs: See below
            mail_options (List): Mail options (see smtplib documentation)
            rcpt_options (List): Recipient options (see smtplib documentation)

        Returns:
            None
        """
        if not users:
            raise ValueError('No users supplied')
        recipients = list()
        for user in users:
            recipients.append(user.data['email'])
        if configuration is None:
            configuration = users[0].configuration
        configuration.emailer().send(recipients, subject, text_body, html_body=html_body, sender=sender, **kwargs)
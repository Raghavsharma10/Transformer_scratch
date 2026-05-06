def _construct_email(self, email, **extra):
        """
        Converts incoming data to properly structured dictionary.
        """
        if isinstance(email, dict):
            email = Email(manager=self._manager, **email)
        elif isinstance(email, (MIMEText, MIMEMultipart)):
            email = Email.from_mime(email, self._manager)
        elif not isinstance(email, Email):
            raise ValueError
        email._update(extra)
        return email.as_dict()
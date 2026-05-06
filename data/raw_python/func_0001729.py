def send(self, recipients, subject, text_body, html_body=None, sender=None, **kwargs):
        # type: (List[str], str, str, Optional[str], Optional[str], Any) -> None
        """
        Send email

        Args:
            recipients (List[str]): Email recipient
            subject (str): Email subject
            text_body (str): Plain text email body
            html_body (Optional[str]): HTML email body
            sender (Optional[str]): Email sender. Defaults to global sender.
            **kwargs: See below
            mail_options (list): Mail options (see smtplib documentation)
            rcpt_options (list): Recipient options (see smtplib documentation)

        Returns:
            None

        """
        if sender is None:
            sender = self.sender
        v = validate_email(sender, check_deliverability=False)  # validate and get info
        sender = v['email']  # replace with normalized form
        normalised_recipients = list()
        for recipient in recipients:
            v = validate_email(recipient, check_deliverability=True)  # validate and get info
            normalised_recipients.append(v['email'])  # replace with normalized form

        if html_body is not None:
            msg = MIMEMultipart('alternative')
            part1 = MIMEText(text_body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)
        else:
            msg = MIMEText(text_body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = ', '.join(normalised_recipients)

        # Perform operations via server
        self.connect()
        self.server.sendmail(sender, normalised_recipients, msg.as_string(), **kwargs)
        self.close()
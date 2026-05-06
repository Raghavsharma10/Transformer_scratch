def send_email(self,
            from_address=None,
            to_address=None,
            subject=None,
            text_data=None,
            html_data=None
    ):
        """Sends an email.

        from_addr is an email address; to_addrs is a list of email adresses.
        Addresses can be plain (e.g. "jsmith@example.com") or with real names
        (e.g. "John Smith <jsmith@example.com>").

        text_data and html_data are both strings.  You can specify one or both.
        If you specify both, the email will be sent as a MIME multipart
        alternative, i.e., the recipient will see the HTML content if his
        viewer supports it; otherwise he'll see the text content.
        """

        settings = self.settings

        from_address = coalesce(from_address, settings["from"], settings.from_address)
        to_address = listwrap(coalesce(to_address, settings.to_address, settings.to_addrs))

        if not from_address or not to_address:
            raise Exception("Both from_addr and to_addrs must be specified")
        if not text_data and not html_data:
            raise Exception("Must specify either text_data or html_data")

        if not html_data:
            msg = MIMEText(text_data)
        elif not text_data:
            msg = MIMEText(html_data, 'html')
        else:
            msg = MIMEMultipart('alternative')
            msg.attach(MIMEText(text_data, 'plain'))
            msg.attach(MIMEText(html_data, 'html'))

        msg['Subject'] = coalesce(subject, settings.subject)
        msg['From'] = from_address
        msg['To'] = ', '.join(to_address)

        if self.server:
            # CALL AS PART OF A SMTP SESSION
            self.server.sendmail(from_address, to_address, msg.as_string())
        else:
            # CALL AS STAND-ALONE
            with self:
                self.server.sendmail(from_address, to_address, msg.as_string())
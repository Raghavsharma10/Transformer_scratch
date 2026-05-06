def from_mime(cls, message, manager):
        """
        Instantiates ``Email`` instance from ``MIMEText`` instance.

        :param message: ``email.mime.text.MIMEText`` instance.
        :param manager: :py:class:`EmailManager` instance.
        :return: :py:class:`Email`
        """
        text, html, attachments = deconstruct_multipart(message)
        subject = prepare_header(message["Subject"])
        sender = prepare_header(message["From"])
        to = prepare_header(message["To"])
        cc = prepare_header(message["Cc"])
        bcc = prepare_header(message["Bcc"])
        reply_to = prepare_header(message["Reply-To"])
        tag = getattr(message, "tag", None)
        return cls(
            manager=manager,
            From=sender,
            To=to,
            TextBody=text,
            HtmlBody=html,
            Subject=subject,
            Cc=cc,
            Bcc=bcc,
            ReplyTo=reply_to,
            Attachments=attachments,
            Tag=tag,
        )
def send(
        self,
        message=None,
        From=None,
        To=None,
        Cc=None,
        Bcc=None,
        Subject=None,
        Tag=None,
        HtmlBody=None,
        TextBody=None,
        Metadata=None,
        ReplyTo=None,
        Headers=None,
        TrackOpens=None,
        TrackLinks="None",
        Attachments=None,
    ):
        """
        Sends a single email.

        :param message: :py:class:`Email` or ``email.mime.text.MIMEText`` instance.
        :param str From: The sender email address.
        :param To: Recipient's email address.
                   Multiple recipients could be specified as a list or string with comma separated values.
        :type To: str or list
        :param Cc: Cc recipient's email address.
                   Multiple Cc recipients could be specified as a list or string with comma separated values.
        :type Cc: str or list
        :param Bcc: Bcc recipient's email address.
                    Multiple Bcc recipients could be specified as a list or string with comma separated values.
        :type Bcc: str or list
        :param str Subject: Email subject.
        :param str Tag: Email tag.
        :param str HtmlBody: HTML email message.
        :param str TextBody: Plain text email message.
        :param str ReplyTo: Reply To override email address.
        :param dict Headers: Dictionary of custom headers to include.
        :param bool TrackOpens: Activate open tracking for this email.
        :param str TrackLinks: Activate link tracking for links in the HTML or Text bodies of this email.
        :param list Attachments: List of attachments.
        :return: Information about sent email.
        :rtype: `dict`
        """
        assert not (message and (From or To)), "You should specify either message or From and To parameters"
        assert TrackLinks in ("None", "HtmlAndText", "HtmlOnly", "TextOnly")
        if message is None:
            message = self.Email(
                From=From,
                To=To,
                Cc=Cc,
                Bcc=Bcc,
                Subject=Subject,
                Tag=Tag,
                HtmlBody=HtmlBody,
                TextBody=TextBody,
                Metadata=Metadata,
                ReplyTo=ReplyTo,
                Headers=Headers,
                TrackOpens=TrackOpens,
                TrackLinks=TrackLinks,
                Attachments=Attachments,
            )
        elif isinstance(message, (MIMEText, MIMEMultipart)):
            message = Email.from_mime(message, self)
        elif not isinstance(message, Email):
            raise TypeError("message should be either Email or MIMEText or MIMEMultipart instance")
        return message.send()
def Email(
        self,
        From,
        To,
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
        Constructs :py:class:`Email` instance.

        :return: :py:class:`Email`
        """
        return Email(
            manager=self,
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
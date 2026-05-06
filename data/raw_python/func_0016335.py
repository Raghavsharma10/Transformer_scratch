def EmailTemplate(
        self,
        TemplateId,
        TemplateModel,
        From,
        To,
        TemplateAlias=None,
        Cc=None,
        Bcc=None,
        Subject=None,
        Tag=None,
        ReplyTo=None,
        Headers=None,
        TrackOpens=None,
        TrackLinks="None",
        Attachments=None,
        InlineCss=True,
    ):
        """
        Constructs :py:class:`EmailTemplate` instance.

        :return: :py:class:`EmailTemplate`
        """
        return EmailTemplate(
            manager=self,
            TemplateId=TemplateId,
            TemplateAlias=TemplateAlias,
            TemplateModel=TemplateModel,
            From=From,
            To=To,
            Cc=Cc,
            Bcc=Bcc,
            Subject=Subject,
            Tag=Tag,
            ReplyTo=ReplyTo,
            Headers=Headers,
            TrackOpens=TrackOpens,
            TrackLinks=TrackLinks,
            Attachments=Attachments,
            InlineCss=InlineCss,
        )
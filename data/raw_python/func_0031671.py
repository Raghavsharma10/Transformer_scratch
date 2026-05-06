def save_model(self, request, obj, form, change):
        """
        sends the email and does not save it
        """
        email = message.EmailMessage(
            subject=obj.subject,
            body=obj.body,
            from_email=obj.from_email,
            to=[t.strip() for t in obj.to_emails.split(',')],
            bcc=[t.strip() for t in obj.bcc_emails.split(',')],
            cc=[t.strip() for t in obj.cc_emails.split(',')]
        )
        email.send()
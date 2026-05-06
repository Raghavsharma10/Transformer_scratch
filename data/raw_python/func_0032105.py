def send_template(self, template, to, reply_to=None, **context):
        """
        Send email from template
        """
        mail_data = self.parse_template(template, **context)
        subject = mail_data["subject"]
        body = mail_data["body"]
        del(mail_data["subject"])
        del(mail_data["body"])

        return self.send(to=to,
                         subject=subject,
                         body=body,
                         reply_to=reply_to,
                         **mail_data)
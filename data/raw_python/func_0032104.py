def send(self, to, subject, body, reply_to=None, **kwargs):
        """
        Send email via AWS SES.
        :returns string: message id

        ***

        Composes an email message based on input data, and then immediately
        queues the message for sending.

        :type to: list of strings or string
        :param to: The To: field(s) of the message.

        :type subject: string
        :param subject: The subject of the message: A short summary of the
                        content, which will appear in the recipient's inbox.

        :type body: string
        :param body: The message body.

        :sender: email address of the sender. String or typle(name, email)
        :reply_to: email to reply to

        **kwargs:

        :type cc_addresses: list of strings or string
        :param cc_addresses: The CC: field(s) of the message.

        :type bcc_addresses: list of strings or string
        :param bcc_addresses: The BCC: field(s) of the message.

        :type format: string
        :param format: The format of the message's body, must be either "text"
                       or "html".

        :type return_path: string
        :param return_path: The email address to which bounce notifications are
                            to be forwarded. If the message cannot be delivered
                            to the recipient, then an error message will be
                            returned from the recipient's ISP; this message
                            will then be forwarded to the email address
                            specified by the ReturnPath parameter.

        :type text_body: string
        :param text_body: The text body to send with this email.

        :type html_body: string
        :param html_body: The html body to send with this email.

        """
        if not self.sender:
            raise AttributeError("Sender email 'sender' or 'source' is not provided")

        kwargs["to_addresses"] = to
        kwargs["subject"] = subject
        kwargs["body"] = body
        kwargs["source"] = self._get_sender(self.sender)[0]
        kwargs["reply_addresses"] = self._get_sender(reply_to or self.reply_to)[2]

        response = self.ses.send_email(**kwargs)
        return response["SendEmailResponse"]["SendEmailResult"]["MessageId"]
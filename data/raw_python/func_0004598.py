def parse(self, request):
        """Parse incoming request and return an email instance.

        Args:
            request: an HttpRequest object, containing a list of forwarded emails, as
                per Mandrill specification for inbound emails.

        Returns:
            a list of EmailMultiAlternatives instances
        """
        assert isinstance(request, HttpRequest), "Invalid request type: %s" % type(request)

        if settings.INBOUND_MANDRILL_AUTHENTICATION_KEY:
            _check_mandrill_signature(
                request=request,
                key=settings.INBOUND_MANDRILL_AUTHENTICATION_KEY,
            )

        try:
            messages = json.loads(request.POST['mandrill_events'])
        except (ValueError, KeyError) as ex:
            raise RequestParseError("Request is not a valid json: %s" % ex)

        if not messages:
            logger.debug("No messages found in mandrill request: %s", request.body)
            return []

        emails = []
        for message in messages:
            if message.get('event') != 'inbound':
                logger.debug("Discarding non-inbound message")
                continue

            msg = message.get('msg')
            try:
                from_email = msg['from_email']
                to = list(self._get_recipients(msg['to']))
                cc = list(self._get_recipients(msg['cc'])) if 'cc' in msg else []
                bcc = list(self._get_recipients(msg['bcc'])) if 'bcc' in msg else []

                subject = msg.get('subject', "")

                attachments = msg.get('attachments', {})
                attachments.update(msg.get('images', {}))

                text = msg.get('text', "")
                html = msg.get('html', "")
            except (KeyError, ValueError) as ex:
                raise RequestParseError(
                    "Inbound request is missing or got an invalid value.: %s." % ex
                )

            email = EmailMultiAlternatives(
                subject=subject,
                body=text,
                from_email=self._get_sender(
                    from_email=from_email,
                    from_name=msg.get('from_name'),
                ),
                to=to,
                cc=cc,
                bcc=bcc,
            )
            if html is not None and len(html) > 0:
                email.attach_alternative(html, "text/html")

            email = self._process_attachments(email, attachments)
            emails.append(email)

        return emails
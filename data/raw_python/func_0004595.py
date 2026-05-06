def parse(self, request):
        """Parse incoming request and return an email instance.

        Args:
            request: an HttpRequest object, containing the forwarded email, as
                per the SendGrid specification for inbound emails.

        Returns:
            an EmailMultiAlternatives instance, containing the parsed contents
                of the inbound email.

        TODO: non-UTF8 charset handling.
        TODO: handler headers.
        """
        assert isinstance(request, HttpRequest), "Invalid request type: %s" % type(request)

        try:
            # from_email should never be a list (unless we change our API)
            from_email = self._get_addresses([_decode_POST_value(request, 'from')])[0]

            # ...but all these can and will be a list
            to_email = self._get_addresses([_decode_POST_value(request, 'to')])
            cc = self._get_addresses([_decode_POST_value(request, 'cc', default='')])
            bcc = self._get_addresses([_decode_POST_value(request, 'bcc', default='')])

            subject = _decode_POST_value(request, 'subject')
            text = _decode_POST_value(request, 'text', default='')
            html = _decode_POST_value(request, 'html', default='')

        except IndexError as ex:
            raise RequestParseError(
                "Inbound request lacks a valid from address: %s." % request.get('from')
            )

        except MultiValueDictKeyError as ex:
            raise RequestParseError("Inbound request is missing required value: %s." % ex)

        if "@" not in from_email:
            # Light sanity check for potential issues related to taking just the
            # first element of the 'from' address list
            raise RequestParseError("Could not get a valid from address out of: %s." % request)

        email = EmailMultiAlternatives(
            subject=subject,
            body=text,
            from_email=from_email,
            to=to_email,
            cc=cc,
            bcc=bcc,
        )
        if html is not None and len(html) > 0:
            email.attach_alternative(html, "text/html")

        # TODO: this won't cope with big files - should really read in in chunks
        for n, f in list(request.FILES.items()):
            if f.size > self.max_file_size:
                logger.debug(
                    "File attachment %s is too large to process (%sB)",
                    f.name,
                    f.size
                )
                raise AttachmentTooLargeError(
                    email=email,
                    filename=f.name,
                    size=f.size
                )
            else:
                email.attach(f.name, f.read(), f.content_type)
        return email
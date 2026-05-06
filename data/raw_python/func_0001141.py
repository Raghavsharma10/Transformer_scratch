def send_mail(
    subject,
    sender,
    to,
    message,
    html_message=None,
    cc=None,
    bcc=None,
    attachments=None,
    host=None,
    port=None,
    auth_user=None,
    auth_password=None,
    use_tls=False,
    fail_silently=False,
):
    """Send a single email to a recipient list.

    All members of the recipient list will see the other recipients in the 'To'
    field.

    Note: The API for this method is frozen. New code wanting to extend the
    functionality should use the EmailMessage class directly.
    """
    if message is None and html_message is None:
        raise ValueError("Either message or html_message must be provided")
    if message is None:
        message = strip_tags(html_message)
    connection = SMTPConnection(
        host=host,
        port=port,
        username=auth_user,
        password=auth_password,
        use_tls=use_tls,
        fail_silently=fail_silently,
    )
    # Convert the to field just for easier usage
    if isinstance(to, six.string_types):
        to = [to]
    if html_message is None:
        email = EmailMessage(
            subject=subject,
            body=message,
            sender=sender,
            to=to,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            connection=connection,
        )
    else:
        email = EmailMultiAlternatives(
            subject=subject,
            body=message,
            sender=sender,
            to=to,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            connection=connection,
        )
        email.attach_alternative(html_message, "text/html")
    return email.send()
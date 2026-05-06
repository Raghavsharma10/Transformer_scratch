def send_mass_mail(
    datatuple, fail_silently=False, auth_user=None, auth_password=None
):
    """Send multiple emails to multiple recipients.

    Given a datatuple of (subject, message, sender, recipient_list), sends
    each message to each recipient list. Returns the number of e-mails sent.

    If auth_user and auth_password are set, they're used to log in.

    Note: The API for this method is frozen. New code wanting to extend the
    functionality should use the EmailMessage class directly.
    """
    connection = SMTPConnection(
        username=auth_user, password=auth_password, fail_silently=fail_silently
    )
    messages = [
        EmailMessage(subject, message, sender, recipient)
        for subject, message, sender, recipient in datatuple
    ]
    return connection.send_messages(messages)
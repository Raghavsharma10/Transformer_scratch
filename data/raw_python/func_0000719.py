def email(sender=None, receivers=(), cc=(), bcc=(),
          subject=None, content=None, encoding='utf8',
          attachments=()):
    """
    Creates a Collection object with a HTML *content*,
    and *attachments*.

    :param content: HTML content.
    :param encoding: Encoding of the email.
    :param attachments: List of filenames to
        attach to the email.
    """
    enclosure = [HTML(content, encoding)]
    enclosure.extend(Attachment(k) for k in attachments)
    return Collection(
        *enclosure,
        headers=[
            headers.subject(subject),
            headers.sender(sender),
            headers.to(*receivers),
            headers.cc(*cc),
            headers.bcc(*bcc),
            headers.date(),
            headers.message_id(),
        ]
    )
def mailto(to, cc=None, bcc=None, subject=None, body=None):
    """
    Generate and run mailto.

    :type to: string
    :param to: The recipient email address.

    :type cc: string
    :param cc: The recipient to copy to.

    :type bcc: string
    :param bcc: The recipient to blind copy to.

    :type subject: string
    :param subject: The subject to use.

    :type body: string
    :param body: The body content to use.
    """

    mailurl = 'mailto:' + str(to)
    if cc is None and bcc is None and subject is None and body is None:
        return str(mailurl)
    mailurl += '?'
    if cc is not None:
        mailurl += 'cc=' + str(cc)
        added = True
    added = False
    if bcc is not None:
        if added is True:
            mailurl += '&'
        mailurl += 'bcc=' + str(cc)
        added = True
    if subject is not None:
        if added is True:
            mailurl += '&'
        mailurl += 'subject=' + str(subject)
        added = True
    if body is not None:
        if added is True:
            mailurl += '&'
        mailurl += 'body=' + str(body)
        added = True
    return mailurl
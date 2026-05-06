def ngettext(singular, plural, n, domain=DEFAULT_DOMAIN):
    """Mark a message as translateable, and translate it considering plural forms.

    Some messages may need to change based on a number. For example, consider a message
    like the following:

    .. code-block:: python

        def alert_msg(msg_count): print(
        'You have %d %s' % (msg_count, 'message' if msg_count == 1 else 'messages'))

        alert_msg(1)
        'You have 1 message'
        alert_msg(5)
        'You have 5 messages'

    To translate this message, you can use ngettext to consider the plural forms:

    .. code-block:: python

        from zengine.lib.translation import ngettext
        def alert_msg(msg_count): print(ngettext('You have %(count)d message',
                                                 'You have %(count)d messages',
                                                 msg_count) % {'count': msg_count})
        alert_msg(1)
        '1 mesajınız var'

        alert_msg(5)
        '5 mesajlarınız var'

    When doing formatting, both singular and plural forms of the message should
    have the exactly same variables.

    Args:
        singular (unicode): The singular form of the message.
        plural (unicode): The plural form of the message.
        n (int): The number that is used to decide which form should be used.
        domain (basestring): The domain of the message. Defaults to 'messages', which
            is the domain where all application messages should be located.
    Returns:
        unicode: The correct pluralization, translated.

    """

    if six.PY2:
        return InstalledLocale._active_catalogs[domain].ungettext(singular, plural, n)
    else:
        return InstalledLocale._active_catalogs[domain].ngettext(singular, plural, n)
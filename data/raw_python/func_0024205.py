def ngettext_lazy(singular, plural, n, domain=DEFAULT_DOMAIN):
    """Mark a message with plural forms translateable, and delay the translation
    until the message is used.

    Works the same was a `ngettext`, with a delaying functionality similiar to `gettext_lazy`.

    Args:
        singular (unicode): The singular form of the message.
        plural (unicode): The plural form of the message.
        n (int): The number that is used to decide which form should be used.
        domain (basestring): The domain of the message. Defaults to 'messages', which
                             is the domain where all application messages should be located.
    Returns:
        unicode: The correct pluralization, with the translation being
                 delayed until the message is used.
    """
    return LazyProxy(ngettext, singular, plural, n, domain=domain, enable_cache=False)
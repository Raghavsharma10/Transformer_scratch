def gettext(message, domain=DEFAULT_DOMAIN):
    """Mark a message as translateable, and translate it.

    All messages in the application that are translateable should be wrapped with this function.
    When importing this function, it should be renamed to '_'. For example:

    .. code-block:: python

        from zengine.lib.translation import gettext as _
        print(_('Hello, world!'))
        'Merhaba, dünya!'

    For the messages that will be formatted later on, instead of using the position-based
    formatting, key-based formatting should be used. This gives the translator an idea what
    the variables in the format are going to be, and makes it possible for the translator
    to reorder the variables. For example:

    .. code-block:: python

        name, number = 'Elizabeth', 'II'
        _('Queen %(name)s %(number)s') % {'name': name, 'number': number}
        'Kraliçe II. Elizabeth'

    The message returned by this function depends on the language of the current user.
    If this function is called before a language is installed (which is normally done
    by ZEngine when the user connects), this function will simply return the message
    without modification.

    If there are messages containing unicode characters, in Python 2 these messages must
    be marked as unicode. Otherwise, python will not be able to correctly match these
    messages with translations. For example:

    .. code-block:: python

        print(_('Café'))
        'Café'
        print(_(u'Café'))
        'Kahve'

    Args:
        message (basestring, unicode): The input message.
        domain (basestring): The domain of the message. Defaults to 'messages', which
            is the domain where all application messages should be located.

    Returns:
        unicode: The translated message.
    """

    if six.PY2:
        return InstalledLocale._active_catalogs[domain].ugettext(message)
    else:
        return InstalledLocale._active_catalogs[domain].gettext(message)
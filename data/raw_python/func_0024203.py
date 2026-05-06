def gettext_lazy(message, domain=DEFAULT_DOMAIN):
    """Mark a message as translatable, but delay the translation until the message is used.

    Sometimes, there are some messages that need to be translated, but the translation
    can't be done at the point the message itself is written. For example, the names of
    the fields in a Model can't be translated at the point they are written, otherwise
    the translation would be done when the file is imported, long before a user even connects.
    To avoid this, `gettext_lazy` should be used. For example:


    .. code-block:: python

        from zengine.lib.translation import gettext_lazy, InstalledLocale
        from pyoko import model, fields
        class User(model.Model):
             name = fields.String(gettext_lazy('User Name'))
        print(User.name.title)
        'User Name'
        
        InstalledLocale.install_language('tr')
        print(User.name.title)
        'Kullanıcı Adı'

    Args:
        message (basestring, unicode): The input message.
        domain (basestring): The domain of the message. Defaults to 'messages', which
            is the domain where all application messages should be located.
    Returns:
        unicode: The translated message, with the translation itself being delayed until
            the text is actually used.

    """
    return LazyProxy(gettext, message, domain=domain, enable_cache=False)
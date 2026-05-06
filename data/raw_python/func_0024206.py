def _wrap_locale_formatter(fn, locale_type):
    """Wrap a Babel data formatting function to automatically format
    for currently installed locale."""

    def wrapped_locale_formatter(*args, **kwargs):
        """A Babel formatting function, wrapped to automatically use the
        currently installed language.

        The wrapped function will not throw any exceptions for unknown locales,
        if Babel doesn't recognise the locale, we will simply fall back to
        the default language.

        The locale used by the wrapped function can be overriden by passing it a `locale` keyword.
        To learn more about this function, check the documentation of Babel for the function of
        the same name.
        """
        # Get the current locale from the class
        kwargs_ = {'locale': getattr(InstalledLocale, locale_type)}
        # By creating a dict then updating it, we allow locale to be overridden
        kwargs_.update(kwargs)
        try:
            formatted = fn(*args, **kwargs_)
        except UnknownLocaleError:
            log.warning(
                """Can\'t do formatting for language code {locale},
                           falling back to default {default}""".format(
                    locale=kwargs_['locale'],
                    default=settings.DEFAULT_LANG)
            )
            kwargs_['locale'] = settings.DEFAULT_LANG
            formatted = fn(*args, **kwargs_)
        return formatted

    return wrapped_locale_formatter
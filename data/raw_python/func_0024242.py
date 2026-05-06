def switch_lang(self):
        """Switch to the language of the current user.

        If the current language is already the specified one, nothing will be done.
        """
        locale = self.current.locale
        translation.InstalledLocale.install_language(locale['locale_language'])
        translation.InstalledLocale.install_locale(locale['locale_datetime'], 'datetime')
        translation.InstalledLocale.install_locale(locale['locale_number'], 'number')
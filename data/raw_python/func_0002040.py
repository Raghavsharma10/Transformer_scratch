def _set_lang_settings(self, lang_settings):
        """ Checks and sets the per language WPM, singular and plural values.
        """
        is_int = isinstance(lang_settings, int)
        is_dict = isinstance(lang_settings, dict)
        if not is_int and not is_dict:
            raise TypeError(("Settings 'READTIME_WPM' must be either an int,"
                             "or a dict with settings per language."))

        # For backwards compatability reasons we'll allow the
        # READTIME_WPM setting to be set as an to override just the default
        # set WPM.
        if is_int:
            self.lang_settings['default']['wpm'] = lang_settings
        elif is_dict:
            for lang, conf in lang_settings.items():
                if 'wpm' not in conf:
                    raise KeyError(('Missing wpm value for the'
                                    'language: {}'.format(lang)))

                if not isinstance(conf['wpm'], int):
                    raise TypeError(('WPM is not an integer for'
                                     ' the language: {}'.format(lang)))

                if "min_singular" not in conf:
                    raise KeyError(('Missing singular form for "minute" for'
                                    ' the language: {}'.format(lang)))

                if "min_plural" not in conf:
                    raise KeyError(('Missing plural form for "minutes" for'
                                    ' the language: {}'.format(lang)))

                if "sec_singular" not in conf:
                    raise KeyError(('Missing singular form for "second" for'
                                    ' the language: {}'.format(lang)))

                if "sec_plural" not in conf:
                    raise KeyError(('Missing plural form for "seconds" for'
                                    ' the language: {}'.format(lang)))

            self.lang_settings = lang_settings
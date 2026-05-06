def translated(self, *language_codes):
        """
        .. versionadded:: 1.0

        Only return translated objects which of the given languages.

        When no language codes are given, only the currently active language is returned.
        """
        # this API has the same semantics as django-parler's .translated() for familiarity.
        # However, since this package doesn't filter in a related field, the ORM limitations don't apply.
        if not language_codes:
            language_codes = (get_language(),)
        else:
            # Since some code operates on a True/str switch, make sure that doesn't drip into this low level code.
            for language_code in language_codes:
                if not isinstance(language_code, six.string_types) or language_code.lower() in ('1', '0', 'true', 'false'):
                    raise ValueError("ContentItemQuerySet.translated() expected language_code to be an ISO code")

        if len(language_codes) == 1:
            return self.filter(language_code=language_codes[0])
        else:
            return self.filter(language_code__in=language_codes)
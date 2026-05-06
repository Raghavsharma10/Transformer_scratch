def collect_translations(self):
        """Collect all `domain` translations and return `Tuple[languages, locale_data]`"""
        languages = {}
        locale_data = {}

        for language_code, label in settings.LANGUAGES:
            languages[language_code] = '%s' % label

            # Create django translation engine for `language_code`
            trans_cat, trans_fallback_cat = self.get_catalog(language_code)

            # Add the meta object
            locale_data[language_code] = {}
            locale_data[language_code][""] = self.make_header(language_code, trans_cat)
            num_plurals = self._num_plurals(trans_cat)

            # Next code is largely taken from Django@master (01.10.2017) from `django.views.i18n JavaScriptCatalogue`
            pdict = {}
            seen_keys = set()

            for key, value in itertools.chain(six.iteritems(trans_cat), six.iteritems(trans_fallback_cat)):
                if key == '' or key in seen_keys:
                    continue

                if isinstance(key, six.string_types):
                    locale_data[language_code][key] = [value]

                elif isinstance(key, tuple):
                    msgid, cnt = key
                    pdict.setdefault(msgid, {})[cnt] = value

                else:
                    raise TypeError(key)
                seen_keys.add(key)

            for k, v in pdict.items():
                locale_data[language_code][k] = [v.get(i, '') for i in range(num_plurals)]

        for key, value in locale_data.items():
            locale_data[key] = json.dumps(value)

        return languages, locale_data
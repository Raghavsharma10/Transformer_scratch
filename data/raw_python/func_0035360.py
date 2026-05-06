def get_best_language(self, accept_lang):
        """Given an Accept-Language header, return the best-matching language."""
        LUM = settings.LANGUAGE_URL_MAP
        langs = dict(LUM.items() + settings.CANONICAL_LOCALES.items())
        # Add missing short locales to the list. This will automatically map
        # en to en-GB (not en-US), es to es-AR (not es-ES), etc. in alphabetical
        # order. To override this behavior, explicitly define a preferred locale
        # map with the CANONICAL_LOCALES setting.
        langs.update((k.split('-')[0], v) for k, v in LUM.items() if
                     k.split('-')[0] not in langs)
        try:
            ranked = parse_accept_lang_header(accept_lang)
        except ValueError:  # see https://code.djangoproject.com/ticket/21078
            return
        else:
            for lang, _ in ranked:
                lang = lang.lower()
                if lang in langs:
                    return langs[lang]
                pre = lang.split('-')[0]
                if pre in langs:
                    return langs[pre]
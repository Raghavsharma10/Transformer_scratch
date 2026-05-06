def get_value(self, context, obj, field_name):
        """
        gets the translated value of field name. If `FALLBACK`evaluates to `True` and the field
        has no translation for the current language, it tries to find a fallback value, using
        the languages defined in `settings.LANGUAGES`.

        """
        try:
            language = get_language()
            value = self.get_translated_value(obj, field_name, language)
            if value:
                return value
            if self.FALLBACK:
                for lang, lang_name in settings.LANGUAGES:
                    if lang == language:
                        # already tried this one...
                        continue
                    value = self.get_translated_value(obj, field_name, lang)
                    if value:
                        return value
            untranslated = getattr(obj, field_name)
            if self._is_truthy(untranslated):
                return untranslated
            else:
                return self.EMPTY_VALUE
        except Exception:
            if settings.TEMPLATE_DEBUG:
                raise
            return self.EMPTY_VALUE
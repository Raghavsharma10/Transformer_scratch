def set_language(self, request, org):
        """Set the current language from the org configuration."""
        if org:
            lang = org.language or settings.DEFAULT_LANGUAGE
            translation.activate(lang)
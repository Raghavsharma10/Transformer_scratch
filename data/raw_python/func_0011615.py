def get_catalog(self, locale):
        """Create Django translation catalogue for `locale`."""
        with translation.override(locale):
            translation_engine = DjangoTranslation(locale, domain=self.domain, localedirs=self.paths)

            trans_cat = translation_engine._catalog
            trans_fallback_cat = translation_engine._fallback._catalog if translation_engine._fallback else {}

            return trans_cat, trans_fallback_cat
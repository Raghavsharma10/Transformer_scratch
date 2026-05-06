def create_title(self, page):
        """
        Create page title in all other languages with cms.api.create_title()
        """
        for language_code, lang_name in iter_languages(self.languages):
            try:
                title = Title.objects.get(page=page, language=language_code)
            except Title.DoesNotExist:
                slug = self.get_slug(language_code, lang_name)
                assert slug != "", "No slug for %r" % language_code
                title = create_title(
                    language=language_code,
                    title=self.get_title(language_code, lang_name),
                    page=page,
                    slug=slug,
                )
                log.debug("Title created: %s", title)
            else:
                log.debug("Page title exist: %s", title)
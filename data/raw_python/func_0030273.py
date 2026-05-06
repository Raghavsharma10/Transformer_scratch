def get_slug(self, language_code, lang_name):
        """
        Notes:
            - slug must be unique!
            - slug is used to check if page already exists!
        :return: 'slug' string for cms.api.create_page()
        """
        title = self.get_title(language_code, lang_name)
        assert title != ""

        title = str(title)  # e.g.: evaluate a lazy translation

        slug = slugify(title)
        assert slug != "", "Title %r results in empty slug!" % title
        return slug
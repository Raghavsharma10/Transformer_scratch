def add_plugins(self, page, placeholder):
        """
        Add a "TextPlugin" in all languages.
        """
        for language_code, lang_name in iter_languages(self.languages):
            for no in range(1, self.dummy_text_count + 1):
                add_plugin_kwargs = self.get_add_plugin_kwargs(
                    page, no, placeholder, language_code, lang_name)

                log.info(
                    'add plugin to placeholder "%s" (pk:%i) in: %s - no: %i',
                    placeholder, placeholder.pk, lang_name, no)
                plugin = add_plugin(
                    placeholder=placeholder,
                    language=language_code,
                    **add_plugin_kwargs)
                log.info('Plugin "%s" (pk:%r) added.', str(plugin), plugin.pk)
                placeholder.save()
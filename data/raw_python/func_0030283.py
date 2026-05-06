def get_title(self, language_code, lang_name):
        """
        :return: 'title' string for cms.api.create_page()
        """
        title = "%s %i-%i in %s" % (self.title_prefix, self.current_count,
                                    self.current_level, language_code)
        log.info(title)
        return title
def publish(self, page):
        """
        Publish the page in all languages.
        """
        assert page.publisher_is_draft == True, "Page '%s' must be a draft!" % page
        publish_page(page, languages=self.languages)
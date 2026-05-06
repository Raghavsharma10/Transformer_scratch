def fill_content(self, page, placeholder_slot):
        """
        Add a placeholder to the page.
        Here we add a "TextPlugin" in all languages.
        """
        if len(placeholder_slot) == 1:
            raise RuntimeError(placeholder_slot)
        placeholder, created = self.get_or_create_placeholder(
            page, placeholder_slot)
        self.add_plugins(page, placeholder)
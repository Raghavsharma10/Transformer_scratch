def get_title(self, plural=True):
        """
        Get's the title of the bundle. Titles can be singular
        or plural.
        """
        value = self.title
        if value == self.parent_attr:
            return self.parent.get_title(plural=plural)

        if not value and self._meta.model:
            value = helpers.model_name(self._meta.model,
                                       self._meta.custom_model_name,
                                       self._meta.custom_model_name_plural,
                                       plural)
        elif self.title and plural:
            value = helpers.pluralize(self.title, self.title_plural)

        return helpers.capfirst_if_needed(value)
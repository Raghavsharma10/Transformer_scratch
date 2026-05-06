def formfield(self, **kwargs):
        """
        Returns a :class:`PlaceholderFormField` instance for this database Field.
        """
        defaults = {
            'label': capfirst(self.verbose_name),
            'help_text': self.help_text,
            'required': not self.blank,
        }
        defaults.update(kwargs)
        return PlaceholderFormField(slot=self.slot, plugins=self._plugins, **defaults)
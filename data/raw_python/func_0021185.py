def active_aliases(self):
        """Get a filtered list of aliases based on configuration.

        Returns aliases and their mappings that are defined in the
        `SEARCH_MAPPINGS` config variable. If the `SEARCH_MAPPINGS` is set to
        `None` (the default), all aliases are included.
        """
        whitelisted_aliases = self.app.config.get('SEARCH_MAPPINGS')
        if whitelisted_aliases is None:
            return self.aliases
        else:
            return {k: v for k, v in self.aliases.items()
                    if k in whitelisted_aliases}
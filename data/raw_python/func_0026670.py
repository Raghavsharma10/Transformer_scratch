def find_entry_name_of_alias(self, alias):
        """Return the first entry name with the given 'alias' included in its
        list of aliases.

        Returns
        -------
        name of matching entry (str) or 'None' if no matches

        """
        if alias in self.aliases:
            name = self.aliases[alias]
            if name in self.entries:
                return name
            else:
                # Name wasn't found, possibly merged or deleted. Now look
                # really hard.
                for name, entry in self.entries.items():
                    aliases = entry.get_aliases(includename=False)
                    if alias in aliases:
                        if (ENTRY.DISTINCT_FROM not in entry or
                                alias not in entry[ENTRY.DISTINCT_FROM]):
                            return name

        return None
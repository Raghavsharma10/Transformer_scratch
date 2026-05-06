def fuzzy_search(self, *filters):
        """
        Perform a "fuzzy" search that matches the given characters in the given order.

        :param filters: The pattern(s) to search for.
        :returns: The matched password names (a list of strings).
        """
        matches = []
        logger.verbose(
            "Performing fuzzy search on %s (%s) ..", pluralize(len(filters), "pattern"), concatenate(map(repr, filters))
        )
        patterns = list(map(create_fuzzy_pattern, filters))
        for entry in self.filtered_entries:
            if all(p.search(entry.name) for p in patterns):
                matches.append(entry)
        logger.log(
            logging.INFO if matches else logging.VERBOSE,
            "Matched %s using fuzzy search.",
            pluralize(len(matches), "password"),
        )
        return matches
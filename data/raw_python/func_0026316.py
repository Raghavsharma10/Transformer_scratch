def simple_search(self, *keywords):
        """
        Perform a simple search for case insensitive substring matches.

        :param keywords: The string(s) to search for.
        :returns: The matched password names (a generator of strings).

        Only passwords whose names matches *all*  of the given keywords are
        returned.
        """
        matches = []
        keywords = [kw.lower() for kw in keywords]
        logger.verbose(
            "Performing simple search on %s (%s) ..",
            pluralize(len(keywords), "keyword"),
            concatenate(map(repr, keywords)),
        )
        for entry in self.filtered_entries:
            normalized = entry.name.lower()
            if all(kw in normalized for kw in keywords):
                matches.append(entry)
        logger.log(
            logging.INFO if matches else logging.VERBOSE,
            "Matched %s using simple search.",
            pluralize(len(matches), "password"),
        )
        return matches
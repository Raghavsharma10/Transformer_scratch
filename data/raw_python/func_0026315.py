def select_entry(self, *arguments):
        """
        Select a password from the available choices.

        :param arguments: Refer to :func:`smart_search()`.
        :returns: The name of a password (a string) or :data:`None`
                  (when no password matched the given `arguments`).
        """
        matches = self.smart_search(*arguments)
        if len(matches) > 1:
            logger.info("More than one match, prompting for choice ..")
            labels = [entry.name for entry in matches]
            return matches[labels.index(prompt_for_choice(labels))]
        else:
            logger.info("Matched one entry: %s", matches[0].name)
            return matches[0]
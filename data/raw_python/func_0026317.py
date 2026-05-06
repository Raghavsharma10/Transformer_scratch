def smart_search(self, *arguments):
        """
        Perform a smart search on the given keywords or patterns.

        :param arguments: The keywords or patterns to search for.
        :returns: The matched password names (a list of strings).
        :raises: The following exceptions can be raised:

                 - :exc:`.NoMatchingPasswordError` when no matching passwords are found.
                 - :exc:`.EmptyPasswordStoreError` when the password store is empty.

        This method first tries :func:`simple_search()` and if that doesn't
        produce any matches it will fall back to :func:`fuzzy_search()`. If no
        matches are found an exception is raised (see above).
        """
        matches = self.simple_search(*arguments)
        if not matches:
            logger.verbose("Falling back from substring search to fuzzy search ..")
            matches = self.fuzzy_search(*arguments)
        if not matches:
            if len(self.filtered_entries) > 0:
                raise NoMatchingPasswordError(
                    format("No passwords matched the given arguments! (%s)", concatenate(map(repr, arguments)))
                )
            else:
                msg = "You don't have any passwords yet! (no *.gpg files found)"
                raise EmptyPasswordStoreError(msg)
        return matches
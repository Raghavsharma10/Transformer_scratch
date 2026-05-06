def _add_tags(self, tags):
        # type: (List[str]) -> bool
        """Add a list of tag

        Args:
            tags (List[str]): list of tags to add

        Returns:
            bool: True if all tags added or False if any already present.
        """
        alltagsadded = True
        for tag in tags:
            if not self._add_tag(tag):
                alltagsadded = False
        return alltagsadded
def remove_section(self, name):
        """Remove a file section.

        Args:
            name: name of the section

        Returns:
            bool: whether the section was actually removed
        """
        existed = self.has_section(name)
        if existed:
            idx = self._get_section_idx(name)
            del self._structure[idx]
        return existed
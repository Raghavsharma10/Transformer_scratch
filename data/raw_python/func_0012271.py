def add_space(self, line):
        """Add a Space object to the section

        Used during initial parsing mainly

        Args:
            line (str): one line that defines the space, maybe whitespaces
        """
        if not isinstance(self.last_item, Space):
            space = Space(self._structure)
            self._structure.append(space)
        self.last_item.add_line(line)
        return self
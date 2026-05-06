def add_comment(self, line):
        """Add a Comment object to the section

        Used during initial parsing mainly

        Args:
            line (str): one line in the comment
        """
        if not isinstance(self.last_item, Comment):
            comment = Comment(self._structure)
            self._structure.append(comment)
        self.last_item.add_line(line)
        return self
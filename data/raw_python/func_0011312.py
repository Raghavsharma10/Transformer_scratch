def top_level_doc(self):
        """The top-level documentation string for the program.
        """
        return self._doc_template.format(
            available_commands='\n  '.join(sorted(self._commands)),
            program=self.program)
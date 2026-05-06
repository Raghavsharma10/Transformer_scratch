def determine_indentation(self):
        """Reset indentation for current token and in self.result to be consistent and normalized"""
        # Ensuring NEWLINE tokens are actually specified as such
        if self.current.tokenum != NEWLINE and self.current.value == '\n':
            self.current.tokenum = NEWLINE

        # I want to change dedents into indents, because they seem to screw nesting up
        if self.current.tokenum == DEDENT:
            self.current.tokenum, self.current.value = self.convert_dedent()

        if self.after_space and not self.is_space and (not self.in_container or self.just_started_container):
            # Record current indentation level
            if not self.indent_amounts or self.current.scol > self.indent_amounts[-1]:
                self.indent_amounts.append(self.current.scol)

            # Adjust indent as necessary
            while self.adjust_indent_at:
                self.result[self.adjust_indent_at.pop()] = (INDENT, self.indent_type * (self.current.scol - self.groups.level))

        # Roll back groups as necessary
        if not self.is_space and not self.in_container:
            while not self.groups.root and self.groups.level >= self.current.scol:
                self.finish_hanging()
                self.groups = self.groups.parent

        # Reset indentation to deal with nesting
        if self.current.tokenum == INDENT and not self.groups.root:
           self.current.value = self.current.value[self.groups.level:]
def next_token(self, tokenum, value, scol):
        """Determine what to do with the next token"""

        # Make self.current reflect these values
        self.current.set(tokenum, value, scol)

        # Determine indent_type based on this token
        if self.current.tokenum == INDENT and self.current.value:
            self.indent_type = self.current.value[0]

        # Only proceed if we shouldn't ignore this token
        if not self.ignore_token():
            # Determining if this token is whitespace
            self.determine_if_whitespace()

            # Determine if inside a container
            self.determine_inside_container()

            # Change indentation as necessary
            self.determine_indentation()

            # See if we are force inserting this token
            if self.forced_insert():
                return

            # If we have a newline after an inserted line, then we don't need to worry about semicolons
            if self.inserted_line and self.current.tokenum == NEWLINE:
                self.inserted_line = False

            # If we have a non space, non comment after an inserted line, then insert a semicolon
            if self.result and not self.is_space and self.inserted_line:
                if self.current.tokenum != COMMENT:
                    self.result.append((OP, ';'))
                self.inserted_line = False

            # Progress the tracker
            self.progress()

            # Add a newline if we just skipped a single
            if self.single and self.single.skipped:
                self.single.skipped = False
                self.result.append((NEWLINE, '\n'))

            # Set after_space so next line knows if it is after space
            self.after_space = self.is_space
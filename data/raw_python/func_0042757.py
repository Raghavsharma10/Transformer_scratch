def forced_insert(self):
        """
            Insert tokens if self.insert_till hasn't been reached yet
            Will respect self.inserted_line and make sure token is inserted before it
            Returns True if it appends anything or if it reached the insert_till token
        """
        # If we have any tokens we are waiting for
        if self.insert_till:
            # Determine where to append this token
            append_at = -1
            if self.inserted_line:
                append_at = -self.inserted_line+1

            # Reset insert_till if we found it
            if self.current.tokenum == self.insert_till[0] and self.current.value == self.insert_till[1]:
                self.insert_till = None
            else:
                # Adjust self.adjust_indent_at to take into account the new token
                for index, value in enumerate(self.adjust_indent_at):
                    if value < len(self.result) - append_at:
                        self.adjust_indent_at[index] = value + 1

                # Insert the new token
                self.result.insert(append_at, (self.current.tokenum, self.current.value))

            # We appended the token
            return True
def determine_if_whitespace(self):
        """
            Set is_space if current token is whitespace
            Is space if value is:
             * Newline
             * Empty String
             * Something that matches regexes['whitespace']
        """
        value = self.current.value

        if value == '\n':
            self.is_space = True
        else:
            self.is_space = False
            if (value == '' or regexes['whitespace'].match(value)):
                self.is_space = True
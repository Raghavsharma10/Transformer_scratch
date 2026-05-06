def write(self, text, add_p_style=True, add_t_style=True):
        """
        see mixed content
        http://effbot.org/zone/element-infoset.htm#mixed-content
        Writing is complicated by requirements of odp to ignore
        duplicate spaces together.  Deal with this by splitting on
        white spaces then dealing with the '' (empty strings) which
        would be the extra spaces
        """
        self._add_styles(add_p_style, add_t_style)
        self._add_pending_nodes()

        spaces = []
        for i, letter in enumerate(text):
            if letter == " ":
                spaces.append(letter)
                continue

            elif len(spaces) == 1:
                self._write(" ")
                self._write(letter)
                spaces = []
                continue

            elif spaces:
                num_spaces = len(spaces) - 1
                # write just a plain space at the start
                self._write(" ")
                if num_spaces > 1:
                    # write the attrib only if more than one space
                    self.add_node("text:s", {"text:c": str(num_spaces)})
                else:
                    self.add_node("text:s")
                self.pop_node()
                self._write(letter)
                spaces = []
                continue

            self._write(letter)

        if spaces:
            num_spaces = len(spaces)
            if num_spaces > 1:
                self.add_node("text:s", {"text:c": str(num_spaces)})
            else:
                self.add_node("text:s")
            self.pop_node()
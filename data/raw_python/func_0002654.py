def wrap(self, text):
        """wrap(text : string) -> [string]

        Reformat the multiple paragraphs in 'text' so they fit in lines of
        no more than 'self.width' columns, and return a list of wrapped
        lines.  Tabs in 'text' are expanded with string.expandtabs(),
        and all other whitespace characters (including newline) are
        converted to space.
        """
        lines = []

        linewrap = partial(textwrap.TextWrapper.wrap, self)
        for para in self.split(text):
            lines.extend(linewrap(para))
            lines.append('')    # Add newline between paragraphs

        # Remove trailing newline
        lines = lines[:-1]

        return lines
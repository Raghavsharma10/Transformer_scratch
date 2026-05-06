def parse(self, text):
        """Parse the given code and add it to :attr:`scripts`.

        The syntax matches :attr:`Script.stringify()`. See :mod:`kurt.text` for
        reference.

        """
        self.scripts.append(kurt.text.parse(text, self))
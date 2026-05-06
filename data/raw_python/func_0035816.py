def feed(self, text=None, source=None):
        """
        Feed some text to the database, either from a string (``text``) or a
        file (``source``).

        >>> db = TrigramsDB()
        >>> db.feed("This is my text")
        >>> db.feed(source="some/file.txt")
        """
        if text is not None:
            words = re.split(r'\s+', text)
            wlen = len(words)
            for i in range(wlen - 2):
                self._insert(words[i:i+3])

        if source is not None:
            with open(source, 'r') as f:
                self.feed(f.read())
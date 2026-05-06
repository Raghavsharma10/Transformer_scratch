def generate(self, **kwargs):
        """
        Generate some text from the database. By default only 70 words are
        generated, but you can change this using keyword arguments.

        Keyword arguments:

            - ``wlen``: maximum length (words)
            - ``words``: a list of words to use to begin the text with
        """
        words = list(map(self._sanitize, kwargs.get('words', [])))
        max_wlen = kwargs.get('wlen', 70)

        wlen = len(words)

        if wlen < 2:
            if not self._db:
                return ''

            if wlen == 0:
                words = sample(self._db.keys(), 1)[0].split(self._WSEP)
            elif wlen == 1:
                spl = [k for k in self._db.keys()
                       if k.startswith(words[0]+self._WSEP)]
                words.append(sample(spl, 1)[0].split(self._WSEP)[1])

            wlen = 2

        while wlen < max_wlen:
            next_word = self._get(words[-2], words[-1])
            if next_word is None:
                break

            words.append(next_word)
            wlen += 1

        return ' '.join(words)
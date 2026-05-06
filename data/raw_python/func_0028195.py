def stem(self, words, parser, **kwargs):
        """
        Get stems for the words using a given parser

        Example:
            from .parsing import ListParser

            parser = ListParser()
            stemmer = Morfologik()
            stemmer.stem(['ja tańczę a ona śpi], parser)

            [
                 ('ja': ['ja']),
                 ('tańczę': ['tańczyć']),
                 ('a': ['a']),
                 ('ona': ['on']),
                 ('śpi': ['spać'])
            ]
        """
        output = self._run_morfologik(words)
        return parser.parse(output, **kwargs)
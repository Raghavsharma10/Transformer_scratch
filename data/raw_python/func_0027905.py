def load_stopwords(self, path):

        """
        Load a set of stopwords.

        Args:
            path (str): The stopwords file path.
        """

        if path:
            with open(path) as f:
                self.stopwords = set(f.read().splitlines())

        else:
            self.stopwords = set(
                pkgutil
                .get_data('textplot', 'data/stopwords.txt')
                .decode('utf8')
                .splitlines()
            )
def plot_term_kdes(self, words, **kwargs):

        """
        Plot kernel density estimates for multiple words.

        Args:
            words (list): A list of unstemmed terms.
        """

        stem = PorterStemmer().stem

        for word in words:
            kde = self.kde(stem(word), **kwargs)
            plt.plot(kde)

        plt.show()
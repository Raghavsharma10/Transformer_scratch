def phrase_strings(self, phrase_type):
        """
        Returns strings corresponding all phrases matching a given phrase type

        :param phrase_type: POS such as "NP", "VP", "det", etc.
        :type phrase_type: str

        :return: a list of strings representing those phrases

        """
        return [u" ".join(subtree.leaves()) for subtree in self.subtrees_for_phrase(phrase_type)]
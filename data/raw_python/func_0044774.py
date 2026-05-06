def subtrees_for_phrase(self, phrase_type):
        """
        Returns subtrees corresponding all phrases matching a given phrase type

        :param phrase_type: POS such as "NP", "VP", "det", etc.
        :type phrase_type: str

        :return: a list of NLTK.Tree.Subtree instances
        :rtype: list of NLTK.Tree.Subtree

        """
        return [subtree for subtree in self.parse.subtrees() if subtree.node.lower() == phrase_type.lower()]
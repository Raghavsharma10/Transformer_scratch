def similarity_score(multicolor1, multicolor2):
        """ Computes how similar two :class:`Multicolor` objects are from perspective of information, that they contain.

        Two multicolors are called to be similar if they contain same colors (at least one). Multiplicity of colors is taken into account as well.

        :param multicolor1: first out of two multi-colors to compute similarity between
        :type multicolor1: :class:`Multicolor`
        :param multicolor2: second out of two multi-colors to compute similarity between
        :type multicolor2: :class:`Multicolor`
        :return: the similarity score between two supplied :class:`Multicolor` object
        :rtype: ``int``
        """
        result = 0
        for key, value in multicolor1.multicolors.items():
            if key in multicolor2.multicolors:
                result += min(value, multicolor2.multicolors[key])
        return result
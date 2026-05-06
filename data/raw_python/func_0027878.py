def anchored_pairs(self, anchor):

        """
        Get distances between an anchor term and all other terms.

        Args:
            anchor (str): The anchor term.

        Returns:
            OrderedDict: The distances, in descending order.
        """

        pairs = OrderedDict()

        for term in self.keys:
            score = self.get_pair(anchor, term)
            if score: pairs[term] = score

        return utils.sort_dict(pairs)
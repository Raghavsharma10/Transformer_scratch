def untrain_token(self, word, count):
        """
        Untrains a particular token (decreases the weight/count of it)

        :param word: the token we're going to train
        :type word: str
        :param count: the number of occurances in the sample
        :type count: int
        """
        if word not in self.tokens:
            return

        # If we're trying to untrain more tokens than we have, we end at 0
        count = min(count, self.tokens[word])

        self.tokens[word] -= count
        self.tally -= count
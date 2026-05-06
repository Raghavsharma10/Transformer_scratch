def train_token(self, word, count):
        """
        Trains a particular token (increases the weight/count of it)

        :param word: the token we're going to train
        :type word: str
        :param count: the number of occurances in the sample
        :type count: int
        """
        if word not in self.tokens:
            self.tokens[word] = 0

        self.tokens[word] += count
        self.tally += count
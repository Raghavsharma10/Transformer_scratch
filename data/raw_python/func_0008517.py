def head(self):
        """ Yields the head of the chunk (usually, the last word in the chunk).
        """
        if self.type == "NP" and any(w.type.startswith("NNP") for w in self):
            w = find(lambda w: w.type.startswith("NNP"), reversed(self))
        elif self.type == "NP":  # "the cat" => "cat"
            w = find(lambda w: w.type.startswith("NN"), reversed(self))
        elif self.type == "VP":  # "is watching" => "watching"
            w = find(lambda w: w.type.startswith("VB"), reversed(self))
        elif self.type == "PP":  # "from up on" => "from"
            w = find(lambda w: w.type.startswith(("IN", "PP")), self)
        elif self.type == "PNP": # "from up on the roof" => "roof"
            w = find(lambda w: w.type.startswith("NN"), reversed(self))
        else:
            w = None
        if w is None:
            w = self[-1]
        return w
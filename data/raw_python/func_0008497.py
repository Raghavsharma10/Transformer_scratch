def train(self, s, path="spelling.txt"):
        """ Counts the words in the given string and saves the probabilities at the given path.
            This can be used to generate a new model for the Spelling() constructor.
        """
        model = {}
        for w in re.findall("[a-z]+", s.lower()):
            model[w] = w in model and model[w] + 1 or 1
        model = ("%s %s" % (k, v) for k, v in sorted(model.items()))
        model = "\n".join(model)
        f = open(path, "w")
        f.write(model)
        f.close()
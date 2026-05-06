def _do_conjunction(self, _and=("and", "e", "en", "et", "und", "y")):
        """ Attach conjunctions.
            CC-words like "and" and "or" between two chunks indicate a conjunction.
        """
        w = self.words
        if len(w) > 2 and w[-2].type == "CC" and w[-2].chunk is None:
            cc  = w[-2].string.lower() in _and and AND or OR
            ch1 = w[-3].chunk
            ch2 = w[-1].chunk
            if ch1 is not None and \
               ch2 is not None:
                ch1.conjunctions.append(ch2, cc)
                ch2.conjunctions.append(ch1, cc)
def scan(self, string):
        """ Returns True if search(Sentence(string)) may yield matches.
            If is often faster to scan prior to creating a Sentence and searching it.
        """
        # In the following example, first scan the string for "good" and "bad":
        # p = Pattern.fromstring("good|bad NN")
        # for s in open("parsed.txt"):
        #     if p.scan(s):
        #         s = Sentence(s)
        #         m = p.search(s)
        #         if m:
        #             print(m)
        w = (constraint.words for constraint in self.sequence if not constraint.optional)
        w = itertools.chain(*w)
        w = [w.strip(WILDCARD) for w in w if WILDCARD not in w[1:-1]]
        if w and not any(w in string.lower() for w in w):
            return False
        return True
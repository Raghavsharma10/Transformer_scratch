def assessments(self, words=[], negation=True):
        """ Returns a list of (chunk, polarity, subjectivity, label)-tuples for the given list of words:
            where chunk is a list of successive words: a known word optionally
            preceded by a modifier ("very good") or a negation ("not good").
        """
        a = []
        m = None # Preceding modifier (i.e., adverb or adjective).
        n = None # Preceding negation (e.g., "not beautiful").
        for w, pos in words:
            # Only assess known words, preferably by part-of-speech tag.
            # Including unknown words (polarity 0.0 and subjectivity 0.0) lowers the average.
            if w is None:
                continue
            if w in self and pos in self[w]:
                p, s, i = self[w][pos]
                # Known word not preceded by a modifier ("good").
                if m is None:
                    a.append(dict(w=[w], p=p, s=s, i=i, n=1, x=self.labeler.get(w)))
                # Known word preceded by a modifier ("really good").
                if m is not None:
                    a[-1]["w"].append(w)
                    a[-1]["p"] = max(-1.0, min(p * a[-1]["i"], +1.0))
                    a[-1]["s"] = max(-1.0, min(s * a[-1]["i"], +1.0))
                    a[-1]["i"] = i
                    a[-1]["x"] = self.labeler.get(w)
                # Known word preceded by a negation ("not really good").
                if n is not None:
                    a[-1]["w"].insert(0, n)
                    a[-1]["i"] = 1.0 / a[-1]["i"]
                    a[-1]["n"] = -1
                # Known word may be a negation.
                # Known word may be modifying the next word (i.e., it is a known adverb).
                m = None
                n = None
                if pos and pos in self.modifiers or any(map(self[w].__contains__, self.modifiers)):
                    m = (w, pos)
                if negation and w in self.negations:
                    n = w
            else:
                # Unknown word may be a negation ("not good").
                if negation and w in self.negations:
                    n = w
                # Unknown word. Retain negation across small words ("not a good").
                elif n and len(w.strip("'")) > 1:
                    n = None
                # Unknown word may be a negation preceded by a modifier ("really not good").
                if n is not None and m is not None and (pos in self.modifiers or self.modifier(m[0])):
                    a[-1]["w"].append(n)
                    a[-1]["n"] = -1
                    n = None
                # Unknown word. Retain modifier across small words ("really is a good").
                elif m and len(w) > 2:
                    m = None
                # Exclamation marks boost previous word.
                if w == "!" and len(a) > 0:
                    a[-1]["w"].append("!")
                    a[-1]["p"] = max(-1.0, min(a[-1]["p"] * 1.25, +1.0))
                # Exclamation marks in parentheses indicate sarcasm.
                if w == "(!)":
                    a.append(dict(w=[w], p=0.0, s=1.0, i=1.0, n=1, x=IRONY))
                # EMOTICONS: {("grin", +1.0): set((":-D", ":D"))}
                if w.isalpha() is False and len(w) <= 5 and w not in PUNCTUATION: # speedup
                    for (type, p), e in EMOTICONS.items():
                        if w in map(lambda e: e.lower(), e):
                            a.append(dict(w=[w], p=p, s=1.0, i=1.0, n=1, x=MOOD))
                            break
        for i in range(len(a)):
            w = a[i]["w"]
            p = a[i]["p"]
            s = a[i]["s"]
            n = a[i]["n"]
            x = a[i]["x"]
            # "not good" = slightly bad, "not bad" = slightly good.
            a[i] = (w, p * -0.5 if n < 0 else p, s, x)
        return a
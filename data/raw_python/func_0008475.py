def _v(self, token, previous=None, next=None):
        """ Returns a training vector for the given (word, tag)-tuple and its context.
        """
        def f(v, s1, s2):
            if s2: 
                v[s1 + " " + s2] = 1
        p, n = previous, next
        p = ("", "") if not p else (p[0] or "", p[1] or "")
        n = ("", "") if not n else (n[0] or "", n[1] or "")
        v = {}
        f(v,  "b", "b")         # Bias.
        f(v,  "h", token[0])    # Capitalization.
        f(v,  "w", token[-6:] if token not in self.known or token in self.unknown else "")
        f(v,  "x", token[-3:])  # Word suffix.
        f(v, "-x", p[0][-3:])   # Word suffix left.
        f(v, "+x", n[0][-3:])   # Word suffix right.
        f(v, "-t", p[1])        # Tag left.
        f(v, "-+", p[1] + n[1]) # Tag left + right.
        f(v, "+t", n[1])        # Tag right.
        return v
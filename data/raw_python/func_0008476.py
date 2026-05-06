def apply(self, token, previous=(None, None), next=(None, None)):
        """ Applies lexical rules to the given token, which is a [word, tag] list.
        """
        w = token[0]
        for r in self:
            if r[1] in self._cmd: # Rule = ly hassuf 2 RB x
                f, x, pos, cmd = bool(0), r[0], r[-2], r[1].lower()
            if r[2] in self._cmd: # Rule = NN s fhassuf 1 NNS x
                f, x, pos, cmd = bool(1), r[1], r[-2], r[2].lower().lstrip("f")
            if f and token[1] != r[0]:
                continue
            if (cmd == "word"       and x == w) \
            or (cmd == "char"       and x in w) \
            or (cmd == "haspref"    and w.startswith(x)) \
            or (cmd == "hassuf"     and w.endswith(x)) \
            or (cmd == "addpref"    and x + w in self.known) \
            or (cmd == "addsuf"     and w + x in self.known) \
            or (cmd == "deletepref" and w.startswith(x) and w[len(x):] in self.known) \
            or (cmd == "deletesuf"  and w.endswith(x) and w[:-len(x)] in self.known) \
            or (cmd == "goodleft"   and x == next[0]) \
            or (cmd == "goodright"  and x == previous[0]):
                token[1] = pos
        return token
def apply(self, tokens):
        """ Applies contextual rules to the given list of tokens,
            where each token is a [word, tag] list.
        """
        o = [("STAART", "STAART")] * 3 # Empty delimiters for look ahead/back.
        t = o + tokens + o
        for i, token in enumerate(t):
            for r in self:
                if token[1] == "STAART":
                    continue
                if token[1] != r[0] and r[0] != "*":
                    continue
                cmd, x, y = r[2], r[3], r[4] if len(r) > 4 else ""
                cmd = cmd.lower()
                if (cmd == "prevtag"        and x ==  t[i-1][1]) \
                or (cmd == "nexttag"        and x ==  t[i+1][1]) \
                or (cmd == "prev2tag"       and x ==  t[i-2][1]) \
                or (cmd == "next2tag"       and x ==  t[i+2][1]) \
                or (cmd == "prev1or2tag"    and x in (t[i-1][1], t[i-2][1])) \
                or (cmd == "next1or2tag"    and x in (t[i+1][1], t[i+2][1])) \
                or (cmd == "prev1or2or3tag" and x in (t[i-1][1], t[i-2][1], t[i-3][1])) \
                or (cmd == "next1or2or3tag" and x in (t[i+1][1], t[i+2][1], t[i+3][1])) \
                or (cmd == "surroundtag"    and x ==  t[i-1][1] and y == t[i+1][1]) \
                or (cmd == "curwd"          and x ==  t[i+0][0]) \
                or (cmd == "prevwd"         and x ==  t[i-1][0]) \
                or (cmd == "nextwd"         and x ==  t[i+1][0]) \
                or (cmd == "prev1or2wd"     and x in (t[i-1][0], t[i-2][0])) \
                or (cmd == "next1or2wd"     and x in (t[i+1][0], t[i+2][0])) \
                or (cmd == "prevwdtag"      and x ==  t[i-1][0] and y == t[i-1][1]) \
                or (cmd == "nextwdtag"      and x ==  t[i+1][0] and y == t[i+1][1]) \
                or (cmd == "wdprevtag"      and x ==  t[i-1][1] and y == t[i+0][0]) \
                or (cmd == "wdnexttag"      and x ==  t[i+0][0] and y == t[i+1][1]) \
                or (cmd == "wdand2aft"      and x ==  t[i+0][0] and y == t[i+2][0]) \
                or (cmd == "wdand2tagbfr"   and x ==  t[i-2][1] and y == t[i+0][0]) \
                or (cmd == "wdand2tagaft"   and x ==  t[i+0][0] and y == t[i+2][1]) \
                or (cmd == "lbigram"        and x ==  t[i-1][0] and y == t[i+0][0]) \
                or (cmd == "rbigram"        and x ==  t[i+0][0] and y == t[i+1][0]) \
                or (cmd == "prevbigram"     and x ==  t[i-2][1] and y == t[i-1][1]) \
                or (cmd == "nextbigram"     and x ==  t[i+1][1] and y == t[i+2][1]):
                    t[i] = [t[i][0], r[1]]
        return t[len(o):-len(o)]
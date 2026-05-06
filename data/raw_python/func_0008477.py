def insert(self, i, tag, affix, cmd="hassuf", tagged=None):
        """ Inserts a new rule that assigns the given tag to words with the given affix,
            e.g., Morphology.append("RB", "-ly").
        """
        if affix.startswith("-") and affix.endswith("-"):
            affix, cmd = affix[+1:-1], "char"
        if affix.startswith("-"):
            affix, cmd = affix[+1:-0], "hassuf"
        if affix.endswith("-"):
            affix, cmd = affix[+0:-1], "haspref"
        if tagged:
            r = [tagged, affix, "f"+cmd.lstrip("f"), tag, "x"]
        else:
            r = [affix, cmd.lstrip("f"), tag, "x"]
        lazylist.insert(self, i, r)
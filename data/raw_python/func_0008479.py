def insert(self, i, tag1, tag2, cmd="prevtag", x=None, y=None):
        """ Inserts a new rule that updates words with tag1 to tag2,
            given constraints x and y, e.g., Context.append("TO < NN", "VB")
        """
        if " < " in tag1 and not x and not y:
            tag1, x = tag1.split(" < "); cmd="prevtag"
        if " > " in tag1 and not x and not y:
            x, tag1 = tag1.split(" > "); cmd="nexttag"
        lazylist.insert(self, i, [tag1, tag2, cmd, x or "", y or ""])
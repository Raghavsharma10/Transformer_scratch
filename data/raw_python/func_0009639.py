def parseBranches(self, descendants):
        """
        Parse top level of markdown

        :param list elements: list of source objects
        :return: list of filtered TreeOfContents objects
        """
        parsed, parent, cond = [], False, lambda b: (b.string or '').strip()
        for branch in filter(cond, descendants):
            if self.getHeadingLevel(branch) == self.depth:
                parsed.append({'root':branch.string, 'source':branch})
                parent = True
            elif not parent:
                parsed.append({'root':branch.string, 'source':branch})
            else:
                parsed[-1].setdefault('descendants', []).append(branch)
        return [TOC(depth=self.depth+1, **kwargs) for kwargs in parsed]
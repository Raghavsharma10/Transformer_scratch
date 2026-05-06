def expandDescendants(self, branches):
        """
        Expand descendants from list of branches

        :param list branches: list of immediate children as TreeOfContents objs
        :return: list of all descendants
        """
        return sum([b.descendants() for b in branches], []) + \
            [b.source for b in branches]
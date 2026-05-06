def expand_polytomies(self, 
        map_attr="name", 
        polytomy_size_limit=5,
        skip_large_polytomies=False):
        """
        Given a tree with one or more polytomies, this functions returns the
        list of all trees (in newick format) resulting from the combination of
        all possible solutions of the multifurcated nodes.

        .. warning:

           Please note that the number of of possible binary trees grows
           exponentially with the number and size of polytomies. Using this
           function with large multifurcations is not feasible:

           polytomy size: 3 number of binary trees: 3
           polytomy size: 4 number of binary trees: 15
           polytomy size: 5 number of binary trees: 105
           polytomy size: 6 number of binary trees: 945
           polytomy size: 7 number of binary trees: 10395
           polytomy size: 8 number of binary trees: 135135
           polytomy size: 9 number of binary trees: 2027025

        http://ajmonline.org/2010/darwin.php
        """

        class TipTuple(tuple):
            pass

        def add_leaf(tree, label):
            yield (label, tree)
            if not isinstance(tree, TipTuple) and isinstance(tree, tuple):
                for left in add_leaf(tree[0], label):
                    yield (left, tree[1])
            for right in add_leaf(tree[1], label):
                yield (tree[0], right)

        def enum_unordered(labels):
            if len(labels) == 1:
                yield labels[0]
            else:
                for tree in enum_unordered(labels[1:]):
                    for new_tree in add_leaf(tree, labels[0]):
                        yield new_tree

        n2subtrees = {}
        for n in self.traverse("postorder"):
            if n.is_leaf():
                subtrees = [getattr(n, map_attr)]
            else:
                subtrees = []
                if len(n.children) > polytomy_size_limit:
                    if skip_large_polytomies:
                        for childtrees in itertools.product(*[n2subtrees[ch] for ch in n.children]):
                            subtrees.append(TipTuple(childtrees))
                    else:
                        raise TreeError("Found polytomy larger than current limit: %s" %n)
                else:
                    for childtrees in itertools.product(*[n2subtrees[ch] for ch in n.children]):
                        subtrees.extend([TipTuple(subtree) for subtree in enum_unordered(childtrees)])

            n2subtrees[n] = subtrees
        return ["%s;"%str(nw) for nw in n2subtrees[self]]
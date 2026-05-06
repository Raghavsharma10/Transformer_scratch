def __get_substitution_paths(g):
        """
        get atoms paths from detached atom to attached

        :param g: CGRContainer
        :return: tuple of atoms numbers
        """
        for n, nbrdict in g.adjacency():
            for m, l in combinations(nbrdict, 2):
                nms = nbrdict[m]['sp_bond']
                nls = nbrdict[l]['sp_bond']
                if nms == (1, None) and nls == (None, 1):
                    yield m, n, l
                elif nms == (None, 1) and nls == (1, None):
                    yield l, n, m
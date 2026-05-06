def _prt(self, name, lparent, sep, pre1, pre2):
        """
        Print a row (leaf) of tree.

        :param name: Full node name
        :type  name: string

        :param lparent: Position in full node name of last separator before
                        node to be printed
        :type  lparent: integer

        :param pre1: Connector next to node name, either a null character if
                     the node to print is the root node, a right angle if node
                     name to be printed is a leaf or a rotated "T" if the node
                     name to be printed is one of many children
        :type  pre1: string
        """
        # pylint: disable=R0914
        nname = name[lparent + 1 :]
        children = self._db[name]["children"]
        ncmu = len(children) - 1
        plst1 = ncmu * [self._vertical_and_right] + [self._up_and_right]
        plst2 = ncmu * [self._vertical] + [" "]
        slist = (ncmu + 1) * [sep + pre2]
        dmark = " (*)" if self._db[name]["data"] else ""
        return "\n".join(
            [
                u"{sep}{connector}{name}{dmark}".format(
                    sep=sep, connector=pre1, name=nname, dmark=dmark
                )
            ]
            + [
                self._prt(child, len(name), sep=schar, pre1=p1, pre2=p2)
                for child, p1, p2, schar in zip(children, plst1, plst2, slist)
            ]
        )
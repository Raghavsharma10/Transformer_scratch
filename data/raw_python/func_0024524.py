def environment(self, atom):
        """
        pairs of (bond, atom) connected to atom

        :param atom: number
        :return: list
        """
        return tuple((bond, self._node[n]) for n, bond in self._adj[atom].items())
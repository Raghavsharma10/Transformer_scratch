def implicify_hydrogens(self):
        """
        remove explicit hydrogen if possible

        :return: number of removed hydrogens
        """
        explicit = defaultdict(list)
        c = 0
        for n, atom in self.atoms():
            if atom.element == 'H':
                for m in self.neighbors(n):
                    if self._node[m].element != 'H':
                        explicit[m].append(n)

        for n, h in explicit.items():
            atom = self._node[n]
            len_h = len(h)
            for i in range(len_h, 0, -1):
                hi = h[:i]
                if atom.get_implicit_h([y.order for x, y in self._adj[n].items() if x not in hi]) == i:
                    for x in hi:
                        self.remove_node(x)
                        c += 1
                    break

        self.flush_cache()
        return c
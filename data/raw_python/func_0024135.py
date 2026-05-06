def explicify_hydrogens(self):
        """
        add explicit hydrogens to atoms

        :return: number of added atoms
        """
        tmp = []
        for n, atom in self.atoms():
            if atom.element != 'H':
                for _ in range(atom.get_implicit_h([x.order for x in self._adj[n].values()])):
                    tmp.append(n)
        for n in tmp:
            self.add_bond(n, self.add_atom(H), Bond())

        self.flush_cache()
        return len(tmp)
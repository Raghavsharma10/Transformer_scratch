def centers_list(self):
        """ get a list of lists of atoms of reaction centers
        """
        center = set()
        adj = defaultdict(set)
        for n, atom in self.atoms():
            if atom._reactant != atom._product:
                center.add(n)

        for n, m, bond in self.bonds():
            if bond._reactant != bond._product:
                adj[n].add(m)
                adj[m].add(n)
                center.add(n)
                center.add(m)

        out = []
        while center:
            n = center.pop()
            if n in adj:
                c = set(self.__plain_bfs(adj, n))
                out.append(list(c))
                center.difference_update(c)
            else:
                out.append([n])

        return out
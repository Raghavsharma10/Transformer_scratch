def from_blob(cls, s):
        """Construct a molecular graph from the blob representation"""
        atom_str, edge_str = s.split()
        numbers = np.array([int(s) for s in atom_str.split(",")])
        edges = []
        orders = []
        for s in edge_str.split(","):
            i, j, o = (int(w) for w in s.split("_"))
            edges.append((i, j))
            orders.append(o)
        return cls(edges, numbers, np.array(orders))
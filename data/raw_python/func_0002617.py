def rank_all(self,roots,optimize=False):
        """Computes rank of all vertices.
        add provided roots to rank 0 vertices,
        otherwise update ranking from provided roots.
        The initial rank is based on precedence relationships,
        optimal ranking may be derived from network flow (simplex).
        """
        self._edge_inverter()
        r = [x for x in self.g.sV if (len(x.e_in())==0 and x not in roots)]
        self._rank_init(roots+r)
        if optimize: self._rank_optimize()
        self._edge_inverter()
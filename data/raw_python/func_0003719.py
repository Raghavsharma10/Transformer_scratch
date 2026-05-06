def blob(self):
        """A compact text representation of the graph"""
        atom_str = ",".join(str(number) for number in self.numbers)
        edge_str = ",".join("%i_%i_%i" % (i, j, o) for (i, j), o in zip(self.edges, self.orders))
        return "%s %s" % (atom_str, edge_str)
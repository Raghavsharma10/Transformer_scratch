def get_edge_string(self, i):
        """Return a string based on the bond order"""
        order = self.orders[i]
        if order == 0:
            return Graph.get_edge_string(self, i)
        else:
            # pad with zeros to make sure that string sort is identical to number sort
            return "%03i" % order
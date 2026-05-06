def get_vertex_string(self, i):
        """Return a string based on the atom number"""
        number = self.numbers[i]
        if number == 0:
            return Graph.get_vertex_string(self, i)
        else:
            # pad with zeros to make sure that string sort is identical to number sort
            return "%03i" % number
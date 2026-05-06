def variables(self):
        """Sorted list of strings summarizing all variables handled by the
        |Node| objects"""
        variables = set([])
        for node in self.nodes:
            variables.add(node.variable)
        return sorted(variables)
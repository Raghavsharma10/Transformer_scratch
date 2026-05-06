def variables(self):
        """
        Returns variables in the logical network

        Returns
        -------
        set[str]
            Unique variables names
        """
        variables = set()
        for v in self.nodes_iter():
            if isinstance(v, Clause):
                for l in v:
                    variables.add(l.variable)
            else:
                variables.add(v)
        return variables
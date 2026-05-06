def subset(self, selector):
        """
        Returns a list of atom indices corresponding to a MDTraj DSL
        query. Also will accept list of numbers, which will be coerced
        to int and returned.
        """
        if isinstance(selector, (list, tuple)):
            return map(int, selector)
        selector = SELECTORS.get(selector, selector)
        mdtop = MDTrajTopology.from_openmm(self.handler.topology)
        return mdtop.select(selector)
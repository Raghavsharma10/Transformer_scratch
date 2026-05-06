def relevantindices(self) -> List[int]:
        """A |list| of all currently relevant indices, calculated as an
        intercection of the (constant) class attribute `RELEVANT_VALUES`
        and the (variable) property |IndexMask.refindices|."""
        return [idx for idx in numpy.unique(self.refindices.values)
                if idx in self.RELEVANT_VALUES]
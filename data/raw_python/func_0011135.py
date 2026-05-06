def _calculate_block_structure(self, inequalities, equalities,
                                   momentinequalities, momentequalities,
                                   extramomentmatrix, removeequalities,
                                   block_struct=None):
        """Calculates the block_struct array for the output file.
        """
        block_struct = []
        if self.verbose > 0:
            print("Calculating block structure...")
        block_struct.append(len(self.monomial_sets[0]) *
                            len(self.monomial_sets[1]))
        if extramomentmatrix is not None:
            for _ in extramomentmatrix:
                block_struct.append(len(self.monomial_sets[0]) *
                                    len(self.monomial_sets[1]))
        super(MoroderHierarchy, self).\
            _calculate_block_structure(inequalities, equalities,
                                       momentinequalities, momentequalities,
                                       extramomentmatrix,
                                       removeequalities,
                                       block_struct=block_struct)
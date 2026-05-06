def _calculate_block_structure(self, inequalities, equalities,
                                   momentinequalities, momentequalities,
                                   extramomentmatrix, removeequalities,
                                   block_struct=None):
        """Calculates the block_struct array for the output file.
        """
        super(SteeringHierarchy, self).\
          _calculate_block_structure(inequalities, equalities,
                                     momentinequalities, momentequalities,
                                     extramomentmatrix, removeequalities)
        if self.matrix_var_dim is not None:
            self.block_struct = [self.matrix_var_dim*bs
                                 for bs in self.block_struct]
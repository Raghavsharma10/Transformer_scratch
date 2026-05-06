def coupling(self, source_y, target_y, weight):
        """How to couple the output of one node to the input of another.
        Args:
          source_y (array of shape (8,)): state of the source node
          target_y (array of shape (8,)): state of the target node
          weight (float): the connection strength
        Returns:
          input (array of shape (8,)): value to drive each variable of the
            target node.
        """
        v_pyramidal = source_y[1] - source_y[2]
        return (np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) *
                (weight*self.g1*self.He2*self.ke2*self.S(v_pyramidal)))
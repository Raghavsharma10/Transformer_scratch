def coupling(self, source_y, target_y, weight):
        """How to couple the output of one subsystem to the input of another.

        This is a fallback default coupling function that should usually be
        replaced with your own.

        This example coupling function takes the mean of all variables of the
        source subsystem and uses that value weighted by the connection
        strength to drive all variables of the target subsystem.

        Arguments:
          source_y (array of shape (d,)): State of the source subsystem.
          target_y (array of shape (d,)): State of target subsystem.
          weight (float): the connection strength for this connection.

        Returns:
          input (array of shape (d,)): Values to drive each variable of the
            target system.
        """
        return np.ones_like(target_y)*np.mean(source_y)*weight
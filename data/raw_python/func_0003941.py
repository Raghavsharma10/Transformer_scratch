def configure(self, x0, axis):
        """Configure the 1D function for a line search

           Arguments:
             x0  --  the reference point (q=0)
             axis  --  a unit vector in the direction of the line search
        """
        self.x0 = x0
        self.axis = axis
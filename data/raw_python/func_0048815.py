def validate(self):
        """
        Validates whether the ODE equations provided make sense  i.e. the number of right-hand side equations
        match the number of left-hand side equations.
        """
        if self.left_hand_side.rows != self.right_hand_side.rows:
            raise ValueError("There are {0} left hand side equations and {1} right hand side equations. "
                             "The same number is expected.".format(self.left_hand_side.rows, self.right_hand_side.rows))
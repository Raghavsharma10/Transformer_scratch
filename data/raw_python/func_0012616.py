def _calc_xy(self, xxx_todo_changeme, angle, length):
        """
        Calculates the coordinates after a specific saccade was made.
        
        Parameters:
            (x,y) : tuple of floats or ints
                The coordinates before the saccade was made
            angle : float or int
                The angle that the next saccade encloses with the 
                horizontal display border
            length: float or int
                The length of the next saccade
        """
        (x, y) = xxx_todo_changeme
        return (x+(cos(radians(angle))*length),
                y+(sin(radians(angle))*length))
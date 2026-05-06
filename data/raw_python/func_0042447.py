def forall(self, method):
        """
        IT IS EXPECTED THE method ACCEPTS (value, coord, cube), WHERE
        value - VALUE FOUND AT ELEMENT
        coord - THE COORDINATES OF THE ELEMENT (PLEASE, READ ONLY)
        cube - THE WHOLE CUBE, FOR USE IN WINDOW FUNCTIONS
        """
        for c in self._all_combos():
            method(self[c], c, self.cube)
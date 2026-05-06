def forall(self, method):
        """
        TODO: I AM NOT HAPPY THAT THIS WILL NOT WORK WELL WITH WINDOW FUNCTIONS
        THE parts GIVE NO INDICATION OF NEXT ITEM OR PREVIOUS ITEM LIKE rownum
        DOES.  MAYBE ALGEBRAIC EDGES SHOULD BE LOOPED DIFFERENTLY?  ON THE
        OTHER HAND, MAYBE WINDOW FUNCTIONS ARE RESPONSIBLE FOR THIS COMPLICATION
        MAR 2015: THE ISSUE IS parts, IT SHOULD BE coord INSTEAD

        IT IS EXPECTED THE method ACCEPTS (value, coord, cube), WHERE
        value - VALUE FOUND AT ELEMENT
        parts - THE ONE PART CORRESPONDING TO EACH EDGE
        cube - THE WHOLE CUBE, FOR USE IN WINDOW FUNCTIONS
        """
        if not self.is_value:
            Log.error("Not dealing with this case yet")

        matrix = self.data.values()[0]
        parts = [e.domain.partitions for e in self.edges]
        for c in matrix._all_combos():
            method(matrix[c], [parts[i][cc] for i, cc in enumerate(c)], self)
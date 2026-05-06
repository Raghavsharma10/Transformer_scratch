def calculate2d(self, force=True):
        """
        recalculate 2d coordinates. currently rings can be calculated badly.

        :param force: ignore existing coordinates of atoms
        """
        for ml in (self.__reagents, self.__reactants, self.__products):
            for m in ml:
                m.calculate2d(force)
        self.fix_positions()
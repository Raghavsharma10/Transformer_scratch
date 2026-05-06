def implicify_hydrogens(self):
        """
        remove explicit hydrogens if possible

        :return: number of removed hydrogens
        """
        total = 0
        for ml in (self.__reagents, self.__reactants, self.__products):
            for m in ml:
                if hasattr(m, 'implicify_hydrogens'):
                    total += m.implicify_hydrogens()
        if total:
            self.flush_cache()
        return total
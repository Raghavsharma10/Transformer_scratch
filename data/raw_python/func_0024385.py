def fix_positions(self):
        """
        fix coordinates of molecules in reaction
        """
        shift_x = 0
        for m in self.__reactants:
            max_x = self.__fix_positions(m, shift_x, 0)
            shift_x = max_x + 1
        arrow_min = shift_x

        if self.__reagents:
            for m in self.__reagents:
                max_x = self.__fix_positions(m, shift_x, 1.5)
                shift_x = max_x + 1
        else:
            shift_x += 3
        arrow_max = shift_x - 1

        for m in self.__products:
            max_x = self.__fix_positions(m, shift_x, 0)
            shift_x = max_x + 1
        self._arrow = (arrow_min, arrow_max)
        self.flush_cache()
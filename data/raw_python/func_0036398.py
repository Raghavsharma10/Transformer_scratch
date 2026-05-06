def setting(self):
        """
        Load setting (Amps, Watts, or Ohms depending on program mode)
        """
        prog_type = self.__program.program_type
        return self._setting / self.SETTING_DIVIDES[prog_type]
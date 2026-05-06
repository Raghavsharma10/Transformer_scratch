def add_step(self, setting, duration):
        """
        Adds steps to a program.
        :param setting: Current, Wattage or Resistance, depending on program mode.
        :param duration: Length of step in seconds.
        :return: None
        """
        if len(self._prog_steps) < 10:
            self._prog_steps.append(ProgramStep(self, setting, duration))
        else:
            raise IndexError("Maximum of 10 steps are allowed")
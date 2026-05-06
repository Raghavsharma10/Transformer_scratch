def load_conditions(self) -> None:
        """Save the initial conditions of the |Model| object handled by
        each |Element| object."""
        for element in printtools.progressbar(self):
            element.model.sequences.load_conditions()
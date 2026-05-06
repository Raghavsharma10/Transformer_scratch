def save_conditions(self) -> None:
        """Save the calculated conditions of the |Model| object handled by
        each |Element| object."""
        for element in printtools.progressbar(self):
            element.model.sequences.save_conditions()
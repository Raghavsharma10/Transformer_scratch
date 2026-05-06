def prepare_inputseries(self, ramflag: bool = True) -> None:
        """Call method |Element.prepare_inputseries| of all handled
        |Element| objects."""
        for element in printtools.progressbar(self):
            element.prepare_inputseries(ramflag)
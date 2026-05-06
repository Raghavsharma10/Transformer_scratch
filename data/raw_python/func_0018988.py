def prepare_stateseries(self, ramflag: bool = True) -> None:
        """Call method |Element.prepare_stateseries| of all handled
        |Element| objects."""
        for element in printtools.progressbar(self):
            element.prepare_stateseries(ramflag)
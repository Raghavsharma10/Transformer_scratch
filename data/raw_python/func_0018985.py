def prepare_allseries(self, ramflag: bool = True) -> None:
        """Call method |Element.prepare_allseries| of all handled
        |Element| objects."""
        for element in printtools.progressbar(self):
            element.prepare_allseries(ramflag)
def prepare_fluxseries(self, ramflag: bool = True) -> None:
        """Call method |Element.prepare_fluxseries| of all handled
        |Element| objects."""
        for element in printtools.progressbar(self):
            element.prepare_fluxseries(ramflag)
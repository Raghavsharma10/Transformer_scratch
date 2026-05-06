def prepare_simseries(self, ramflag: bool = True) -> None:
        """Call method |Node.prepare_simseries| of all handled
        |Node| objects."""
        for node in printtools.progressbar(self):
            node.prepare_simseries(ramflag)
def prepare_obsseries(self, ramflag: bool = True) -> None:
        """Call method |Node.prepare_obsseries| of all handled
        |Node| objects."""
        for node in printtools.progressbar(self):
            node.prepare_obsseries(ramflag)
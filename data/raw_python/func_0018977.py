def prepare_allseries(self, ramflag: bool = True) -> None:
        """Call methods |Node.prepare_simseries| and
        |Node.prepare_obsseries|."""
        self.prepare_simseries(ramflag)
        self.prepare_obsseries(ramflag)
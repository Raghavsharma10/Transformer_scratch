def _info(self):
        """Returns an OrderedDict of information, for human display."""
        d = OrderedDict()

        d['vid'] = self.vid
        d['sname'] = self.sname
        d['vname'] = self.vname

        return d
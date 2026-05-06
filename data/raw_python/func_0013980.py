def luns(self):
        """Aggregator for ioclass_luns and ioclass_snapshots."""
        lun_list, smp_list = [], []
        if self.ioclass_luns:
            lun_list = map(lambda l: VNXLun(lun_id=l.lun_id, name=l.name,
                                            cli=self._cli), self.ioclass_luns)
        if self.ioclass_snapshots:
            smp_list = map(lambda smp: VNXLun(name=smp.name, cli=self._cli),
                           self.ioclass_snapshots)
        return list(lun_list) + list(smp_list)
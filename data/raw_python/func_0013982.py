def modify(self, new_name=None, iotype=None, lun_ids=None, smp_names=None,
               ctrlmethod=None, minsize=None, maxsize=None):
        """Overwrite the current properties for a VNX ioclass.

        :param new_name: new name for the ioclass
        :param iotype: can be 'rw', 'r' or 'w'
        :param lun_ids: list of LUN IDs
        :param smp_names: list of Snapshot Mount Point names
        :param ctrlmethod: the new CtrlMethod
        :param minsize: minimal size in kb
        :param maxsize: maximum size in kb
        """
        if not any([new_name, iotype, lun_ids, smp_names, ctrlmethod]):
            raise ValueError('Cannot apply modification, please specify '
                             'parameters to modify.')

        def _do_modify():
            out = self._cli.modify_ioclass(
                self._get_name(), new_name, iotype, lun_ids, smp_names,
                ctrlmethod, minsize, maxsize)
            ex.raise_if_err(out, default=ex.VNXIOClassError)

        try:
            _do_modify()
        except ex.VNXIOCLassRunningError:
            with restart_policy(self.policy):
                _do_modify()

        return VNXIOClass(new_name if new_name else self._get_name(),
                          self._cli)
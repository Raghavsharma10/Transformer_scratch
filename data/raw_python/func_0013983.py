def add_lun(self, luns):
        """A wrapper for modify method.

        .. note:: This API only append luns to existing luns.
        """
        curr_lun_ids, curr_smp_names = self._get_current_names()
        luns = normalize_lun(luns, self._cli)
        new_ids, new_smps = convert_lun(luns)
        if new_ids:
            curr_lun_ids.extend(new_ids)
        if new_smps:
            curr_smp_names.extend(new_smps)
        return self.modify(lun_ids=curr_lun_ids, smp_names=curr_smp_names)
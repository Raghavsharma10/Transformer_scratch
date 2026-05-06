def replace_lun(self, *lun_list):
        """Replaces the exiting LUNs to lun_list."""
        lun_add = self._prepare_luns_add(lun_list)
        lun_remove = self._prepare_luns_remove(lun_list, False)
        return self.modify(lun_add=lun_add, lun_remove=lun_remove)
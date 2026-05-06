def update_lun(self, add_luns=None, remove_luns=None):
        """Updates the LUNs in CG, adding the ones in `add_luns` and removing
        the ones in `remove_luns`"""
        if not add_luns and not remove_luns:
            log.debug("Empty add_luns and remove_luns passed in, "
                      "skip update_lun.")
            return RESP_OK
        lun_add = self._prepare_luns_add(add_luns)
        lun_remove = self._prepare_luns_remove(remove_luns, True)
        return self.modify(lun_add=lun_add, lun_remove=lun_remove)
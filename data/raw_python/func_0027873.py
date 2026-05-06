def remove_all_lvs(self):
        """
        Removes all logical volumes from the volume group.

        *Raises:*

        *       HandleError,  CommitError
        """
        lvs = self.lvscan()
        for lv in lvs:
            self.remove_lv(lv)
def has_hlu(self, lun_or_snap, cg_member=None):
        """Returns True if `lun_or_snap` is attached to the host.

        :param lun_or_snap: can be lun, lun snap, cg snap or a member snap of
            cg snap.
        :param cg_member: the member lun of cg if `lun_or_snap` is cg snap.
        :return: True - if `lun_or_snap` is attached, otherwise False.
        """
        hlu = self.get_hlu(lun_or_snap, cg_member=cg_member)
        return hlu is not None
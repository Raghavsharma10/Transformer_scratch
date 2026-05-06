def get_hlu(self, resource, cg_member=None):
        """Gets the hlu number of a lun, lun snap, cg snap or a member snap of
        cg snap.

        :param resource: can be lun, lun snap, cg snap or a member snap of cg
            snap.
        :param cg_member: the member lun of cg if `lun_or_snap` is cg snap.
        :return: the hlu number.
        """
        host_lun = self.get_host_lun(resource, cg_member=cg_member)
        return host_lun if host_lun is None else host_lun.hlu
def get_host_lun(self, lun_or_snap, cg_member=None):
        """Gets the host lun of a lun, lun snap, cg snap or a member snap of cg
        snap.

        :param lun_or_snap: can be lun, lun snap, cg snap or a member snap of
            cg snap.
        :param cg_member: the member lun of cg if `lun_or_snap` is cg snap.
        :return: the host lun object.
        """
        import storops.unity.resource.lun as lun_module
        import storops.unity.resource.snap as snap_module
        which = None
        if isinstance(lun_or_snap, lun_module.UnityLun):
            which = self._get_host_luns(lun=lun_or_snap)
        elif isinstance(lun_or_snap, snap_module.UnitySnap):
            if lun_or_snap.is_cg_snap():
                if cg_member is None:
                    log.debug('None host lun for CG snap {}. '
                              'Use its member snap instead or pass in '
                              'cg_member.'.format(lun_or_snap.id))
                    return None
                lun_or_snap = lun_or_snap.get_member_snap(cg_member)
                which = self._get_host_luns(lun=cg_member, snap=lun_or_snap)
            else:
                which = self._get_host_luns(snap=lun_or_snap)
        if not which:
            log.debug('Resource(LUN or Snap) {} is not attached to host {}'
                      .format(lun_or_snap.name, self.name))
            return None
        return which[0]
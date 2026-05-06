def has_snap(self):
        """ This method won't count the snaps in "destroying" state!

        :return: false if no snaps or all snaps are destroying.
        """
        return len(list(filter(lambda s: s.state != SnapStateEnum.DESTROYING,
                               self.snapshots))) > 0
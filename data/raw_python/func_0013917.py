def thin_clone(self, name, io_limit_policy=None, description=None):
        """Creates a new thin clone from this snapshot.
        .. note:: this snapshot should not enable Auto-Delete.
        """
        if self.is_member_snap():
            raise UnityCGMemberActionNotSupportError()

        if self.lun and not self.lun.is_thin_enabled:
            raise UnityThinCloneNotAllowedError()

        return TCHelper.thin_clone(self._cli, self, name, io_limit_policy,
                                   description)